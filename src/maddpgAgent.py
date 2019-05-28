from collections import deque
from keras import Model

from keras import backend as K
from tensorflow.python.ops.parallel_for import jacobian
import tensorflow as tf
import numpy as np

import random


class AbstractMaddpgAgent:

    def __init__(self, agent_id, action_dim, observation_dim, learning_rate=0.01):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Should be keras/tensorflow models (Support predict, train)
        # Should be something, one day, maybe it will be something else, dont know yet
        # Maybe it should be a something in between
        self.critic = None
        self.policy = None


        # Private members
        self._state_input = None
        self._other_action_input = None
        self._action_input = None

        self._critic_gradient = None
        self._policy_jacobian = None

        self.policy_opti_ops = None

    def mk_critic_model(self):
        raise NotImplemented

    def mk_policy_model(self):
        raise NotImplemented

    def mk_critic_action_gradient(self):
        self._critic_gradient = K.gradients(self.critic.output, self._action_input)

    def mk_policy_jacobian(self):
        self._policy_jacobian = jacobian(self.policy.output, self.policy.trainable_weights)

    def mk_policy_opt(self):
        self.mk_critic_action_gradient()
        self.mk_policy_jacobian()

        gradJ = tf.reduce_mean(self._policy_jacobian * self._critic_gradient)

        update_ops = []
        for i,w in enumerate(self.policy.trainable_weights):
            update_ops.append(w + self.learning_rate * gradJ[i])

        self.policy_opti_ops = update_ops


    def critic_action_gradient(self, s, a):
        s = K.get_session()
        return s.run([self._critic_gradient], feed_dict={self._state_input: s, self._action_input: a})[0]

    def policy_jacobian(self):
        s = K.get_session()
        return s.run([self._policy_jacobian], feed_dict={self._state_input})[0]

    def act(self, observation, exploration=True):
        action = self.policy.predict(observation)[0]

        return action + self.random_distrib() if exploration else action

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
    def __init__(self, env, nb_agent=3, agent_class = None, memory_size=1000, batch_size=32, gamma=0.95, horizon=None):
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


    def update_critic(self, sample):

        y = [[] for i in range(self.nb_agent)]
        X = [[] for i in range(self.nb_agent)]

        for state, actions, rewards, next_state in sample:

            # First we build the target value y:
            actions = []

            for k in range(self.nb_agent):
                agent = self.agents[k]
                observation = agent.watch(state, k)
                action = agent.act(observation)

                actions.append(action)

            for i in range(self.nb_agent):
                y[i].append(rewards[i] + self.gamma * self.agents[i].Q(state, actions))
                X[i].append(np.asarray([state, actions]))

        for i in range(self.nb_agent):
            self.agents[i].critic.train_on_batch(X[i], y[i])










