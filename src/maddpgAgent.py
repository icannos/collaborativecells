from collections import deque
from keras import Model

from keras import backend as K
from tensorflow.python.ops.parallel_for import jacobian
import tensorflow as tf
import numpy as np

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

        self.critic = None
        self.policy = None
        self.target_policy = None

        self._critic_gradient = None
        self._policy_jacobian = None

        self.policy_opti_ops = None

        self.update_target = [tf.assign(t, tau * e + (1-tau) * t)
                              for t,e in zip(self.target_policy.trainable_weights, self.policy.trainable_weights)]

    def critic_inputs(self, stateoraction, agent):
        idx = 0

        if stateoraction == "action":
            idx += self.nb_agent

        idx += agent

        return self.critic.inputs[idx]

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
        raise NotImplemented

    def watch(self, state, i):
        """
        Take a state and return an observation, ie what this agent actually sees. By default, just returns the ith
        element of state
        :param state: state of the get from the environnment
        :return: observation, what THIS agent sees
        """
        raise NotImplemented

    def random_distrib(self):
        raise NotImplemented


class ReplayBuffer:
    def __init__(self, memory_size, batch_size):
        self.batch_size = batch_size
        self.memory_size = memory_size

        # We store history for each agent
        # Because their rewards/trajectory are not the same
        self.memory = deque([], maxlen=memory_size)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))


class AbstractMaddpgTrainer:
    def __init__(self, env, nb_agent=3, agent_class=None, memory_size=1000, batch_size=32, gamma=0.95, horizon=None):
        self.gamma = gamma
        self.horizon = horizon
        self.env = env
        self.agent_class = agent_class
        self.nb_agent = nb_agent
        self.memory_size = memory_size

        self.action_dim = env.action_space
        self.observation_dim = env.observation_space

        self.buffer = ReplayBuffer(memory_size=memory_size, batch_size=batch_size)

        self.agents = []
        for agent in range(nb_agent):
            self.agents.append(agent_class[agent](self.action_dim, self.observation_dim))

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

