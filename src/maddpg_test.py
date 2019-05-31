"""
author: Maxime Darrin

"""


import numpy as np
import tensorflow as tf

from multiagent.environment import MultiAgentEnv
from maddpgAgent import DenseAgent, AbstractMaddpgTrainer


import multiagent.scenarios as scenarios

from dqlAgents import simpleAgent

from time import time

episodes = 300
episode_duration = 127
steps = 16


# load scenario from script
scenario = scenarios.load("simple" + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment

env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, action_space="continuous")

session = tf.Session()

trainer = AbstractMaddpgTrainer(session, env, 1, [DenseAgent for _ in range(1)])

trainer.train(1000, render=True)