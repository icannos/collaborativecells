"""
author: Maxime Darrin

Main implementation of Deep Q learning algorithm and k-step Deep Q Learning Algorithm
"""


from collections import deque

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

import random
import numpy as np


class DeepQLearner:
    """
    The DeepQLearner for the chrome dino game.
    """

    def __init__(self, action_space, input_shape=(210, 160, 3), mem_size=10000, learning_rate=0.001,
                 exploration_max=1,
                 exploration_min=0.01, exploration_decay=0.995, batch_size=32, gamma=0.99):
        """

        :param action_space:
        :param input_shape:
        :param mem_size:
        :param learning_rate:
        :param exploration_max:
        :param exploration_min:
        :param exploration_decay:
        :param batch_size:
        :param gamma:
        """
        self._memory_size = mem_size

        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.batch_size = batch_size
        self.gamma = gamma

        self._memory = deque(maxlen=mem_size)
        self.action_space = action_space

        self.input_shape = input_shape
        self.learning_rate = learning_rate

        self.build_model()

    def build_model(self):
        raise NotImplemented

    def store_exp(self, state, action, reward, next_state, done):
        """
        Store experiences ie transition state, action --> reward, next_state, done so it can be used in experience
        review
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self._memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose between random action and action from the model.
        :param state:
        :return:
        """
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        else:
            q = self.model.predict(state)[0]
            return np.argmax(q)

    def exp_review(self):
        """
        Train the model using experience review. Takes data from memory and train the network with it.
        :return:
        """
        if len(self._memory) < self.batch_size:
            return

        # Get a batch of data to use
        batch = random.sample(self._memory, self.batch_size)

        # Prepare inputs array
        inputs = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                          dtype=np.float)

        # Prepare outputs array
        targets = np.zeros((self.batch_size, self.action_space))

        # Experience replay
        for i in range(0, self.batch_size):
            state_t = batch[i][0]
            action_t = batch[i][1]
            reward_t = batch[i][2]
            state_t1 = batch[i][3]
            terminal = batch[i][4]

            inputs[i] = state_t[0]

            targets[i] = self.model.predict(state_t)

            Q_sa = self.model.predict(state_t1)

            # Update the gain expectation
            if terminal:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.gamma * np.max(Q_sa)

        # Make a single gradient descent step
        self.model.train_on_batch(inputs, targets)

        # Update exploration ratio
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def large_review(self, steps=10):
        """
        Train the model using experience review. Takes data from memory and train the network with it.
        :return:
        """
        if len(self._memory) < self.batch_size:
            return

        for k in range(steps):
            # Get a batch of data to use
            batch = random.sample(self._memory, self.batch_size)

            # Prepare inputs array
            inputs = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                              dtype=np.float)

            # Prepare outputs array
            targets = np.zeros((self.batch_size, self.action_space))

            # Experience replay
            for i in range(0, self.batch_size):
                state_t = batch[i][0]
                action_t = batch[i][1]
                reward_t = batch[i][2]
                state_t1 = batch[i][3]
                terminal = batch[i][4]

                inputs[i:i + 1] = state_t[0]

                print(self.model.predict(state_t).shape)

                targets[i] = self.model.predict(state_t)[0]

                # print(targets.shape)

                Q_sa = self.model.predict(state_t1)[0]

                # Update the gain expectation
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + self.gamma * np.max(Q_sa)

            # Make a single gradient descent step
            self.model.train_on_batch(inputs, targets)

    def dump(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)


class DeepQLearnerStep(DeepQLearner):
    def store_exp(self, seq):
        """
        Store experiences ie transition state, action --> reward, next_state, done so it can be used in experience
        review
        :return:
        """
        self._memory.append(seq)

    def exp_review(self):
        """
        Train the model using experience review. Takes data from memory and train the network with it.
        :return:
        """
        if len(self._memory) < 50:
            return

        # Get a batch of data to use
        batch = random.sample(self._memory, self.batch_size)

        # Experience replay

        inputs = []
        targets = []

        for i in range(0, self.batch_size):
            seq = batch[i]

            state_t = seq[-1][0]
            action_t = seq[-1][1]
            reward_t = seq[-1][2]
            state_t1 = seq[-1][3]
            terminal = seq[-1][4]

            inputs.append(state_t[0])

            targets.append(self.model.predict(state_t)[0])

            expected_reward = max(self.model.predict(state_t1)[0])

            # Update the gain expectation
            if terminal:
                expected_reward = reward_t
            else:
                expected_reward = reward_t + self.gamma * expected_reward

            targets[-1][action_t] = expected_reward

            for k in range(2, len(seq) + 1):
                state_t, action_t, reward_t, _, terminal = seq[-k]

                inputs.append(state_t[0])

                targets.append(self.model.predict(state_t)[0])

                # Update the gain expectation
                if terminal:
                    expected_reward += reward_t
                else:
                    expected_reward = reward_t + self.gamma * expected_reward

                targets[-1][action_t] = expected_reward

        # Make a single gradient descent step
        self.model.train_on_batch(np.asarray(inputs), np.asarray(targets))

        # Update exploration ratio
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def long_replay(self, steps=10):

        if len(self._memory) < self.batch_size:
            return

        for l in range(steps):
            # Get a batch of data to use
            batch = random.sample(self._memory, self.batch_size)

            # Experience replay

            inputs = []
            targets = []

            for i in range(0, self.batch_size):
                seq = batch[i]

                state_t = seq[-1][0]
                action_t = seq[-1][1]
                reward_t = seq[-1][2]
                state_t1 = seq[-1][3]
                terminal = seq[-1][4]

                inputs.append(state_t[0])

                targets.append(self.model.predict(state_t)[0])

                expected_reward = max(self.model.predict(state_t1)[0])

                # Update the gain expectation
                if terminal:
                    expected_reward = reward_t
                else:
                    expected_reward = reward_t + self.gamma * expected_reward

                targets[-1][action_t] = expected_reward

                for k in range(2, len(seq) + 1):
                    state_t, action_t, reward_t, _, terminal = seq[-k]

                    inputs.append(state_t[0])

                    targets.append(self.model.predict(state_t)[0])

                    # Update the gain expectation
                    if terminal:
                        expected_reward += reward_t
                    else:
                        expected_reward = reward_t + self.gamma * expected_reward

                    targets[-1][action_t] = expected_reward

            # Make a single gradient descent step
            self.model.train_on_batch(np.asarray(inputs), np.asarray(targets))


