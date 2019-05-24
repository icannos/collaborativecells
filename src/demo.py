"""
author: Maxime Darrin

A naive random agent for simple adversary open ai env scenario
"""


import numpy as np

from multiagent.environment import MultiAgentEnv

import multiagent.scenarios as scenarios
from keras.models import load_model

from time import sleep

# load scenario from script
scenario = scenarios.load("simple_adversary" + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment


env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)


nb_agent = len(env.agents)
action_space_n = env.action_space
observation_space_n = env.observation_space


agents = [load_model("../agent-" + str(i)+".model") for i in range(nb_agent)]

while True:
    state_n = env.reset()
    action_space_n = env.action_space
    env.discrete_action_input = True

    state_n = [np.reshape(state_n[i], (1, observation_space_n[i].shape[0])) for i in range(nb_agent)]

    for i in range(100):
        env.render()

        action_n = []
        for i, a in enumerate(agents):
            action = np.argmax(a.predict(state_n[i])[0])
            action_n.append(action)
            sleep(0.01)


        next_state_n, reward_n, done, info = env.step(action_n)

        state_n = [np.reshape(next_state_n[i], (1, observation_space_n[i].shape[0])) for i in range(nb_agent)]
