"""
author: Maxime Darrin

A naive random agent for simple adversary open ai env scenario
"""


import numpy as np

from multiagent.environment import MultiAgentEnv

import multiagent.scenarios as scenarios


# load scenario from script
scenario = scenarios.load("simple_adversary" + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment


env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

while True:
    observation_n = env.reset()
    action_space_n = env.action_space
    env.discrete_action_input = True

    while True:
        env.render()

        action_n = []
        for i in range(len(observation_n)):
            action_n.append(action_space_n[i].sample())

        #observation_n, reward_n, done, info = env.step(action_n)





