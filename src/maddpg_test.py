"""
author: Maxime Darrin

"""


import numpy as np

from multiagent.environment import MultiAgentEnv
from maddpgAgent import DenseAgent, AbstractMaddpgTrainer


import multiagent.scenarios as scenarios

from dqlAgents import simpleAgent

from time import time



episodes = 300
episode_duration = 127
steps = 16


# load scenario from script
scenario = scenarios.load("simple_adversary" + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment

env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
env.discrete_action_input = False

