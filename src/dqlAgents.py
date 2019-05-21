"""
author: Maxime Darrin
Agents powered by usual DQL for simple cell multiagent env from Open AI.
"""
from agent import DeepQLearnerStep

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.initializers import glorot_uniform
from keras.optimizers import Adam


class simpleAgent(DeepQLearnerStep):

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=self.input_shape[0], activation="relu"
                             ))
        self.model.add(Dense(32, activation="relu", use_bias=True
                             ))
        self.model.add(Dense(self.action_space, activation="relu"))

        adam = Adam(lr=self.learning_rate)
        self.model.compile(loss='mse', optimizer=adam)
