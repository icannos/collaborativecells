"""
author: Maxime Darrin

"""

import numpy as np
import tensorflow as tf

from multiagent.environment import MultiAgentEnv
from maddpgAgent import DenseAgent, AbstractMaddpgTrainer
from sys import argv

import multiagent.scenarios as scenarios

from dqlAgents import simpleAgent

from time import time

episodes = 1000
episode_duration = 127
steps = 16

# load scenario from script
scenario = scenarios.load("simple_spread.py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment

env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, action_space="continuous")

session = tf.Session()

trainer = AbstractMaddpgTrainer(session, "trainer", env, 3, [DenseAgent for _ in range(3)])

trainer.train(int(argv[1]), render=False)

trainer.dump_model("model.save")