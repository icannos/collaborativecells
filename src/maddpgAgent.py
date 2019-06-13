from collections import deque
from keras import Model

from keras import backend as K
import tensorflow as tf
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Concatenate, Activation, Input, InputLayer, Flatten
from keras.initializers import glorot_normal, lecun_normal, lecun_uniform
from keras.regularizers import l1

import random


class AbstractMaddpgAgent:
    """
    Abstract class which implement a single agent

    Methods to override:
    --------------------
    mk_critic_model unit --> keras model : should return a keras model object, it is used to build the critic model.
    See the documentation of the method for more informations about the specs of the model
    mk_policy_model unit --> keras model : same as before but for the policy model

    """

    def __init__(self, session, agent_id, action_shapes, observation_shapes, learning_rate=0.01, tau=0.01):
        """

        :param session: A tensorflow session used to run the models and train them
        :param agent_id: the id of this agent among the population of agent in the env
        :param action_shapes: list of shapes, shape of the action space for each agent of the environnemnt
        :param observation_shapes: list of shapes, shape of the observation for each agent of the environnemnt
        :param learning_rate: adam learning rate
        :param tau: update rate of the target model
        """

        if len(observation_shapes) != len(action_shapes):
            raise ValueError

        self.exploration_rate = 0.01
        self.exploration_decay = 0.99

        self.agent_id = agent_id
        self.learning_rate = learning_rate

        self.observation_shapes = observation_shapes
        self.action_shapes = action_shapes

        self.nb_agent = len(observation_shapes)

        self.observations_inputs = [Input(shape) for shape in self.observation_shapes]
        self.actions_inputs = [Input(shape) for shape in self.action_shapes]

        self.critic = self.mk_critic_model()
        self.target_critic = self.mk_critic_model()

        self.policy = self.mk_policy_model()
        self.target_policy = self.mk_policy_model()

        self.optimize_policy = self.mk_policy_opt()

        self.session = session
        K.set_session(session)

        self.update_policy_target = \
            [tf.assign(t, tau * e + (1 - tau) * t)
             for t, e in zip(self.target_policy.trainable_weights, self.policy.trainable_weights)]

        self.update_critic_target = \
            [tf.assign(t, tau * e + (1 - tau) * t)
             for t, e in zip(self.target_critic.trainable_weights, self.critic.trainable_weights)]

    def mk_critic_model(self):
        """
        This class automatically build a list of inputs corresponding to the observations of each agent and to the
        actions of each agent.
        self.observations_inputs
        self.actions_inputs

        You should use those inputs as input for the model you build
        :return: keras model
        """
        raise NotImplemented

    def mk_policy_model(self):
        """
        This class automatically build a list of inputs corresponding to the observations of each agent and to the
        actions of each agent.
        self.observations_inputs
        self.actions_inputs

        You should use those inputs as input for the model you build
        :return: keras model
        """
        raise NotImplemented

    def mk_policy_opt(self):
        """
        Build the tensorflow operation for training the policy, it uses the default inputs created by the class and
        the output of the policy.
        :return: Tensorflow operation (tensor)
        """

        actions_inputs = [self.actions_inputs[i] if i != self.agent_id else self.policy.output
                          for i in range(self.nb_agent)]
        q_i = self.critic(actions_inputs + self.observations_inputs)

        grad = tf.gradients(q_i, self.policy.trainable_weights)

        optimizer = tf.train.AdamOptimizer(-self.learning_rate)

        return optimizer.apply_gradients(zip(grad, self.policy.trainable_weights))

    def act(self, observation, exploration=True):
        """
        Take an action according to the policy and add a random perturbation if exploration is set to true.
        Currently not used. Replaced by explore. but it will eventually will be used again, since "act" is clearer
        than "explore"
        :param observation: observation for this agent and only for this one. It should decide which action to take
        only with its own observations
        :param exploration: bool, if true add random perturbation to the decision
        :return: the action vector taken
        """
        action = self.policy.predict(np.asarray([observation]))[0]

        return action + self.random_distrib() if exploration else action

    def explore(self, observation):
        """
        Take an action according to the policy and add a random perturbation if exploration is set to true.
        :param observation:
        :return:
        """
        action = self.policy.predict(np.asarray([observation]))[0]

        return action + np.random.normal(0, self.exploration_rate, self.action_shapes[self.agent_id])

    def target_action(self, observation):
        """
        Take an action according to the target policy. Without exploration
        :param observation: observation for this agent
        :return: action vector
        """
        action = self.target_policy.predict(np.asarray([observation]))[0]

        return action

    def Q(self, state, actions, model="eval"):
        """
        Make a call to the critic to evaluate an action according to the observations and action from other
        :param state: list of observation (observation for each agent)
        :param actions: list of actions (action for each agent)
        :return: real: the evaluation
        """
        if model == "eval":
            return self.critic.predict([[state[i]] for i in range(self.nb_agent)] +
                                       [[actions[i]] for i in range(self.nb_agent)])[0][0]
        else:
            return self.target_critic.predict([[state[i]] for i in range(self.nb_agent)] +
                                              [[actions[i]] for i in range(self.nb_agent)])[0][0]

    def watch(self, state, i):
        """
        Take a state and return an observation, ie what this agent actually sees. By default, just returns the ith
        element of state
        :param state: state of the get from the environnment
        :return: observation, what THIS agent sees
        """
        return state[i]

    def random_distrib(self):
        """
        This is used for exploration, it should generates a perturbation to add to the action vector
        :return: vector of the same shape as the action vector
        """
        return np.random.normal(0, 0.01, self.action_shapes[self.agent_id])


class ReplayBuffer:
    """
    Simple class to implement the replay buffer.
    """

    def __init__(self, memory_size, batch_size):
        """

        :param memory_size: Number of transition to keep track of
        :param batch_size: Size of a training batch
        """
        self.batch_size = batch_size
        self.memory_size = memory_size

        # The memory itself, it is a queue. When full, adding an element drops an element at the end
        self.memory = deque([], maxlen=memory_size)

    def sample(self):
        """
        Samples randomly a training batch of size batch_size
        :return: a sample of transition
        """
        return random.sample(self.memory, self.batch_size)

    def remember(self, state, action, reward, next_state):
        """
        Store a transition into the replay buffer
        :param state: list of observation at time t (one per agent, the full list is the state of the game)
        :param action: list of action at time t
        :param reward: list of rewards after taking actions at state
        :param next_state: list of the new observations (one per agent, the full list is the next_state of the game)
        :return: None
        """
        self.memory.append((state, action, reward, next_state))


class DenseAgent(AbstractMaddpgAgent):
    """
    A very simple agent which only implement 2 Dense models for critic and policy
    """

    def mk_critic_model(self):
        inputs_states = self.observations_inputs
        inputs_actions = self.actions_inputs

        inputs = inputs_states + inputs_actions

        concat = Concatenate()(inputs_states + inputs_actions)

        layer1 = Dense(64, activation="relu", kernel_initializer=lecun_normal())(concat)
        layer2 = Dense(16, activation="tanh", kernel_initializer=lecun_normal())(layer1)
        layer3 = Dense(1, activation="linear", kernel_initializer=lecun_normal())(layer2)

        model = Model(inputs=inputs, outputs=layer3)
        model.compile(optimizer="adam", loss="mse")

        return model

    def mk_policy_model(self):
        input_shape = self.observation_shapes[self.agent_id]
        output_shape = self.action_shapes[self.agent_id][0]

        model = Sequential()
        model.add(Dense(32, input_shape=input_shape, activation="relu", kernel_initializer=lecun_normal()))
        model.add(Dense(output_shape, activation="tanh", kernel_initializer=lecun_normal()))

        model = Model(inputs=self.observations_inputs[self.agent_id],
                      outputs=model(self.observations_inputs[self.agent_id]))
        model.compile(optimizer="adam", loss="mse")

        return model


class AbstractMaddpgTrainer:
    """
    This class aims to encapsulate the training process of a pool of agents.
    """

    def __init__(self, session, env, nb_agent=3, agent_class=None, memory_size=10 ** 6, batch_size=2048, gamma=0.8,
                 horizon=100):
        """

        :param session: A tensorflow session to do the computations
        :param env: A gym env like object to train on
        :param nb_agent: The number of agents we have to take care of
        :param agent_class: A list of class to build the different agents. (This way every agent can be animed by a
        different model)
        :param memory_size: The size of the replay buffer. Research paper set it to 10**6
        :param batch_size: Size of a training batch. Research paper set it to 1024
        :param gamma: The discount factor
        :param horizon: The duration of a game (if not dead or ended before)
        """
        self.gamma = gamma
        self.horizon = horizon
        self.env = env
        self.agent_class = agent_class
        self.nb_agent = nb_agent
        self.memory_size = memory_size

        self.action_dim = [a.shape for a in env.action_space]
        self.observation_dim = [a.shape for a in env.observation_space]

        self.buffer = ReplayBuffer(memory_size=memory_size, batch_size=batch_size)

        self.session = session
        K.set_session(session)

        self.agents = []
        for agent in range(nb_agent):
            self.agents.append(agent_class[agent](session, agent, self.action_dim, self.observation_dim))

    def train(self, episode=1, render=True):
        """
        Perform the training of the agents
        :param episode: number of game to make
        :param render: bool, Wether we want to display the game or not
        :return: None
        """
        last_train = 0
        last_reward = np.zeros(self.nb_agent)
        for _ in range(episode):
            state = self.env.reset()
            for d in range(self.horizon):
                if render:
                    self.env.render()

                actions = []
                for i in range(self.nb_agent):
                    actions.append(self.agents[i].explore(state[i]))

                next_state, rewards, done, info = self.env.step(actions)

                rewards = np.array(rewards)

                self.buffer.remember(state, actions, np.array(rewards) - last_reward, next_state)
                state = next_state
                last_reward = np.array(rewards)

                if len(self.buffer.memory) < self.buffer.batch_size or last_train < 500:
                    last_train += 1
                    continue

                last_train = 0
                for i in range(self.nb_agent):
                    sample = self.buffer.sample()
                    self.train_step(sample, i)
                    self.agents[i].exploration_rate *= self.agents[i].exploration_decay

                self.update_targets()

    def train_step(self, sample, i):
        """
        Perform a training step of agent i on the sample
        :param sample: sample of transitions to train the agent on
        :param i: The id of the agent to train
        :return: None
        """
        y = []

        states = [[sample[j][0][l] for j in range(len(sample))] for l in range(self.nb_agent)]
        actions = [[sample[j][1][l] for j in range(len(sample))] for l in range(self.nb_agent)]

        s_in = {self.agents[i].observations_inputs[k]: states[k] for k in range(self.nb_agent)}
        a_in = {self.agents[i].actions_inputs[k]: actions[k]
                for k in range(self.nb_agent)}

        s_in_t = {self.agents[i].observations_inputs[k].name.split(":")[0]: np.asarray(states[k]) for k in
                  range(self.nb_agent)}
        a_in_t = {self.agents[i].actions_inputs[k].name.split(":")[0]: np.array(actions[k])
                  for k in range(self.nb_agent)}

        for state, actions, rewards, next_state in sample:

            # First we build the target value y:
            actionsp = []

            for k in range(self.nb_agent):
                agent = self.agents[k]
                observation = agent.watch(state, k)
                action = agent.target_action(observation)

                actionsp.append(action)

            yj = rewards[i] + self.gamma * self.agents[i].Q(next_state, actionsp, "target")
            y.append(yj)

        self.agents[i].critic.fit({**s_in_t, **a_in_t}, y)

        _ = self.session.run([self.agents[i].optimize_policy], feed_dict={**s_in, **a_in})

    def update_targets(self):
        """
        Make a training step of the target model according to target = tau * e + (1-tau) * target
        :return:
        """

        # For each agent, update policy target and critic target using dedicated operations
        for i in range(self.nb_agent):
            self.session.run(self.agents[i].update_policy_target + self.agents[i].update_critic_target)
