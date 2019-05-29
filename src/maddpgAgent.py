from collections import deque
from keras import Model

from keras import backend as K
import tensorflow as tf
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Concatenate, Activation, Input, InputLayer, Flatten

import random


class AbstractMaddpgAgent:

    def __init__(self, agent_id, action_shapes, observation_shapes, learning_rate=0.01, tau=0.01):

        if len(observation_shapes) != len(action_shapes):
            raise ValueError

        self.agent_id = agent_id
        self.learning_rate = learning_rate

        self.observation_shapes = observation_shapes
        self.action_shapes = action_shapes

        self.nb_agent = len(observation_shapes)

        self.observations_inputs = [Input(shape) for shape in self.observation_shapes]
        self.actions_inputs = [Input(shape) for shape in self.action_shapes]

        self.critic = self.mk_critic_model()
        self.policy = self.mk_policy_model()
        self.target_policy = self.mk_policy_model()

        self.optimize_policy = None

        self.update_target = [tf.assign(t, tau * e + (1 - tau) * t)
                              for t, e in zip(self.target_policy.trainable_weights, self.policy.trainable_weights)]

    def mk_critic_model(self):
        raise NotImplemented

    def mk_policy_model(self):
        raise NotImplemented

    def mk_policy_opt(self):
        q_i = self.critic([t if i == self.agent_id + self.nb_agent else self.policy.output
                           for i, t in enumerate(self.critic.inputs)])

        grad = tf.gradients(q_i, self.policy.trainable_weights)

        optimizer = tf.train.AdamOptimizer(-self.learning_rate)

        self.optimize_policy = optimizer.apply_gradients(zip(grad, self.policy.trainable_weights))

    def act(self, observation, exploration=True):
        action = self.policy.predict(observation)[0]

        return action + self.random_distrib() if exploration else action

    def target_action(self, observation):
        action = self.policy.predict(observation)[0]

        return action

    def Q(self, state, actions):
        return self.critic.predict([[state[i]] for i in range(self.nb_agent)] +
                                   [[actions[i]] for i in range(self.nb_agent)])[0]

    def watch(self, state, i):
        """
        Take a state and return an observation, ie what this agent actually sees. By default, just returns the ith
        element of state
        :param state: state of the get from the environnment
        :return: observation, what THIS agent sees
        """
        return state[i]

    def random_distrib(self):
        return np.random.normal(0, 1, self.action_shapes[self.agent_id])


class ReplayBuffer:
    def __init__(self, memory_size, batch_size):
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = deque([], maxlen=memory_size)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))


class DenseAgent(AbstractMaddpgAgent):

    def mk_critic_model(self):
        inputs_states = self.observations_inputs
        inputs_actions = self.actions_inputs

        inputs = inputs_states + inputs_actions

        concat = Concatenate()(inputs_states + inputs_actions)

        print(concat)

        layer1 = Dense(128, activation="relu")(concat)
        layer2 = Dense(16, activation="relu")(layer1)
        layer3 = Dense(self.action_shapes[self.agent_id][0], activation="relu")(layer2)

        model = Model(inputs=inputs, outputs=layer3)
        model = model.compile(optimizer="adam", loss="mse")

    def mk_policy_model(self):
        input_shape = self.observation_shapes[self.agent_id]
        output_shape = self.action_shapes[self.agent_id][0]

        model = Sequential()
        model.add(Dense(32, input_shape=input_shape, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(output_shape, activation="relu"))

        model = Model(inputs=self.observations_inputs[self.agent_id], outputs=model(self.observations_inputs[self.agent_id]))
        model.compile(optimizer="adam", loss="mse")

        return model


class AbstractMaddpgTrainer:
    def __init__(self, env, nb_agent=3, agent_class=None, memory_size=1000, batch_size=32, gamma=0.95, horizon=None):
        self.gamma = gamma
        self.horizon = horizon
        self.env = env
        self.agent_class = agent_class
        self.nb_agent = nb_agent
        self.memory_size = memory_size

        self.action_dim = [a.shape for a in env.action_space]
        self.observation_dim = [a.shape for a in env.observation_space]

        self.buffer = ReplayBuffer(memory_size=memory_size, batch_size=batch_size)

        self.agents = []
        for agent in range(nb_agent):
            self.agents.append(agent_class[agent](agent, self.action_dim, self.observation_dim))

    def train(self, episode=1):

        for _ in range(episode):
            state = self.env.reset()

            while True:
                actions = [self.agents[i].act(state[i]) for i in range(self.nb_agent)]

                next_state, rewards, done, info = self.env.step(actions)

                self.buffer.remember(state, actions, rewards, next_state)

                state = next_state

                for i in range(self.nb_agent):
                    sample = self.buffer.sample()
                    self.train_step(sample, i)

                self.update_targets()

    def train_step(self, sample, i):
        y = []
        X = []

        for state, actions, rewards, next_state in sample:

            # First we build the target value y:
            actionsp = []

            for k in range(self.nb_agent):
                agent = self.agents[k]
                observation = agent.watch(state, k)
                action = agent.target_action(observation)

                actionsp.append(action)

            y.append(rewards[i] + self.gamma * self.agents[i].Q(next_state, actionsp))
            X.append(np.asarray([state, actions]))

        self.agents[i].critic.train_on_batch(X, y)

        states = [[sample[j][0][l] for j in range(len(sample))] for l in range(self.nb_agent)]
        actions = [[sample[j][1][l] for j in range(len(sample))] for l in range(self.nb_agent)]

        s_in = {self.agents[i].critic.inputs[k]: states[k] for k in range(self.nb_agent)}
        a_in = {self.agents[i].critic.inputs[k]: actions[k - self.nb_agent]
                for k in range(self.nb_agent, 2 * self.nb_agent)}

        with K.get_session() as s:
            _ = s.run([self.agents[i].optimize_policy], feed_dict={**s_in, **a_in})

    def update_targets(self):
        with K.get_session() as s:
            for i in range(self.nb_agent):
                s.run(self.agents[i].update_target)
