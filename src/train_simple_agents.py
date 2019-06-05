"""
author: Maxime Darrin

"""


import numpy as np

from multiagent.environment import MultiAgentEnv

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

env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

nb_agent = len(env.agents)
action_space_n = env.action_space
observation_space_n = env.observation_space


agents = [simpleAgent(input_shape=observation_space_n[i].shape, action_space=action_space_n[i].n, batch_size=3)
          for i in range(nb_agent)]

for i in range(episodes):

    print("Episode: ", i)
    state_n = env.reset()

    env.discrete_action_input = True

    step = 0

    seq_n = [[] for i in range(nb_agent)]

    state_n = [np.reshape(state_n[i], (1, observation_space_n[i].shape[0])) for i in range(nb_agent)]
    last_reward = np.zeros(nb_agent)
    while True:
        if step > episode_duration:
            for i in range(nb_agent):
                if seq_n[i] != []:
                    print("================== TRAIN 1 ===============")
                    agents[i].store_exp(seq_n[i])
                    agents[i].exp_review()
            break

        env.render()
        action_n = []

        seq_action_n = [[] for i in range(nb_agent)]

        for i, a in enumerate(agents):
            action = a.act(state_n[i])
            seq_action_n[i].append(action)
            action_n.append(action)

        next_state_n, reward_n, done, info = env.step(action_n)

        next_state_n = [np.reshape(next_state_n[i], (1, observation_space_n[i].shape[0])) for i in range(nb_agent)]

        for i in range(nb_agent):
            if len(seq_n[i]) < steps:
                seq_n[i].append((state_n[i], seq_action_n[i], -last_reward[i] + reward_n[i], next_state_n[i], done[i]))

            else:
                print("================== TRAIN ===============")
                agents[i].store_exp(seq_n[i])
                agents[i].exp_review()
                seq_n[i] = [(state_n[i], seq_action_n[i], reward_n[i], next_state_n[i], done[i])]

        step+=1

        state_n = next_state_n
        print(-last_reward[i] + reward_n[i])
        last_reward = reward_n
        print(reward_n)



for i, a in enumerate(agents):
    a.dump("agent-" + str(i)+".model")




