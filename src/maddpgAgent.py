from collections import deque
import random


class AbstractMaddpgAgent:

    def __init__(self, action_dim, observation_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Should be keras/tensorflow models (Support predict, train)
        self.critic = None
        self.policy = None

    def mk_critic_model(self):
        raise NotImplemented

    def mk_policy_model(self):
        raise NotImplemented

    def act(self, state, exploration=True):
        action = self.policy.predict(state)[0]

        return action + self.random_distrib() if exploration else action

    def evaluate(self, global_state):
        return self.critic.predict(global_state)[0]

    def random_distrib(self):
        raise NotImplemented


class ReplayBuffer:
    def __init__(self, nb_agent, memory_size, batch_size):
        self.batch_size = batch_size
        self.nb_agent = nb_agent
        self.memory_size = memory_size

        # We store history for each agent
        # Because their rewards/trajectory are not the same
        self.memory = [deque([], maxlen=memory_size) for _ in range(nb_agent)]

    def sample(self, agent=None):
        if agent is not None:
            return random.sample(self.memory[agent], self.batch_size)
        else:
            return [random.sample(self.memory[agent], self.batch_size) for agent in range(self.nb_agent)]

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))


class AbstractMaddpgTrainer:
    def __init__(self, env, nb_agent=3, agent_class = None, memory_size=1000, batch_size=32):
        self.agent_class = agent_class
        self.nb_agent = nb_agent
        self.memory_size = memory_size

        self.action_dim = env.action_space
        self.observation_dim = env.observation_space

        self.buffer = ReplayBuffer(nb_agent=nb_agent, memory_size=memory_size, batch_size=batch_size)

        self.agents = []
        for agent in range(nb_agent):
            self.agents.append(agent_class[agent](self.action_dim, self.observation_dim))

    def train(self):
        pass
