import numpy as np
import tensorflow as tf


class Actor:

    def __init__(self, scope, session, n_actions, action_bound,
                 eval_states, target_states, learning_rate=0.001, tau=0.01):
        self.session = session
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.eval_states = eval_states
        self.target_states = target_states
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.eval_actions = self.build_network(self.eval_states,
                                                   scope='eval', trainable=True)
            self.target_actions = self.build_network(self.target_states,
                                                     scope='target', trainable=False)

            self.eval_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=scope + '/eval')
            self.target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope=scope + '/target')

            self.update_target = [tf.assign(t, (1 - tau) * t + tau * e)
                                  for t, e in zip(self.target_weights, self.eval_weights)]

    def build_network(self, x, scope, trainable):
        with tf.variable_scope(scope):
            W = tf.random_normal_initializer(0.0, 0.1)
            b = tf.constant_initializer(0.1)
            h1 = tf.layers.dense(x, 50, activation=tf.nn.relu,
                                 kernel_initializer=W, bias_initializer=b,
                                 name='h1', trainable=trainable)
            actions = tf.layers.dense(h1, self.n_actions, activation=tf.nn.tanh,
                                      kernel_initializer=W, bias_initializer=b,
                                      name='actions', trainable=trainable)
            scaled_actions = tf.multiply(actions, self.action_bound,
                                         name='scaled_actions')

        return scaled_actions

    def add_gradients(self, action_gradients):
        with tf.variable_scope(self.scope):
            self.action_gradients = tf.gradients(ys=self.eval_actions,
                                                 xs=self.eval_weights,
                                                 grad_ys=action_gradients)
            optimizer = tf.train.AdamOptimizer(-self.learning_rate)
            self.optimize = optimizer.apply_gradients(zip(self.action_gradients,
                                                          self.eval_weights))

    def learn(self, actors, states):
        a = {}
        for i in range(len(states)):
            a[actors[i].eval_states] = states[i]

        self.session.run(self.optimize, feed_dict={**a})
        self.session.run(self.update_target)

    def choose_action(self, state):
        return self.session.run(self.eval_actions,
                                feed_dict={self.eval_states: state[np.newaxis, :]})[0]


class Critic:

    def __init__(self, scope, session, n_actions, actors_eval_actions,
                 actors_target_actions, eval_states, target_states,
                 rewards, learning_rate=0.001, gamma=0.9, tau=0.01):
        self.session = session
        self.n_actions = n_actions
        self.actors_eval_actions = actors_eval_actions
        self.actors_target_actions = actors_target_actions
        self.eval_states = eval_states
        self.target_states = target_states
        self.rewards = rewards

        with tf.variable_scope(scope):
            self.eval_values = self.build_network(self.eval_states,
                                                  self.actors_eval_actions,
                                                  'eval', trainable=True)
            self.target_values = self.build_network(self.target_states,
                                                    self.actors_target_actions,
                                                    'target', trainable=False)

            self.eval_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=scope + '/eval')
            self.target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope=scope + '/target')

            self.target = self.rewards + gamma * self.target_values
            self.loss = tf.reduce_mean(tf.squared_difference(self.target,
                                                             self.eval_values))

            self.optimize = tf.train.AdamOptimizer(
                learning_rate).minimize(self.loss)
            self.action_gradients = []
            for i in range(len(self.actors_eval_actions)):
                self.action_gradients.append(tf.gradients(ys=self.eval_values,
                                                          xs=self.actors_eval_actions[i])[0])

            self.update_target = [tf.assign(t, (1 - tau) * t + tau * e)
                                  for t, e in zip(self.target_weights, self.eval_weights)]

    def build_network(self, x1, x2, scope, trainable):
        with tf.variable_scope(scope):
            W = tf.random_normal_initializer(0.0, 0.1)
            b = tf.constant_initializer(0.1)

            first = True
            for i in range(len(x1)):
                h1 = tf.layers.dense(x1[i], 50, activation=tf.nn.relu,
                                     kernel_initializer=W, bias_initializer=b,
                                     name='h1-' + str(i), trainable=trainable)
                h21 = tf.get_variable('h21-' + str(i), [50, 50],
                                      initializer=W, trainable=trainable)
                h22 = tf.get_variable('h22-' + str(i), [self.n_actions[i], 50],
                                      initializer=W, trainable=trainable)

                if first == True:
                    h3 = tf.matmul(h1, h21) + tf.matmul(x2[i], h22)
                    first = False
                else:
                    h3 = h3 + tf.matmul(h1, h21) + tf.matmul(x2[i], h22)

            b2 = tf.get_variable('b2', [1, 50], initializer=b,
                                 trainable=trainable)
            h3 = tf.nn.relu(h3 + b2)
            values = tf.layers.dense(h3, 1, kernel_initializer=W,
                                     bias_initializer=b, name='values',
                                     trainable=trainable)

        return values

    def learn(self, states, actions, rewards, states_next):
        s = {i: d for i, d in zip(self.eval_states, states)}
        a = {i: d for i, d in zip(self.actors_eval_actions, actions)}
        sn = {i: d for i, d in zip(self.target_states, states_next)}

        loss, _ = self.session.run([self.loss, self.optimize], feed_dict={**s, **a, **sn,
                                                                          self.rewards: rewards})
        self.session.run(self.update_target)
        return loss



class AbstractMaddpgTrainer:
    """
    This class aims to encapsulate the training process of a pool of agents.
    """
    def __init__(self, session, env, nb_agent=3, agent_class=None, memory_size=10**6, batch_size=1024, gamma=0.95,
                 horizon=100):
        """

        :param session: A tensorflow session to do the computations
        :param env: A gym env like object to train on
        :param nb_agent: The number of agents we have to take care of
        :param agent_class: A list of class to build the different agents. (This way every agent can be animed by a
        different model)
        :param memory_size: The size of the replay buffer. Research paper set it to 10**6
        :param batch_size: Size of a training batch. Research paper set it to 1024
        :param gamma: The discount factor
        :param horizon: The duration of a game (if not dead or ended before)
        """
        self.gamma = gamma
        self.horizon = horizon
        self.env = env
        self.agent_class = agent_class
        self.nb_agent = nb_agent
        self.memory_size = memory_size

        self.action_dim = [a.shape for a in env.action_space]
        self.observation_dim = [a.shape for a in env.observation_space]

        self.buffer = ReplayBuffer(memory_size=memory_size, batch_size=batch_size)

        self.session = session
        K.set_session(session)

        self.agents = []
        for agent in range(nb_agent):
            self.agents.append(agent_class[agent](session, agent, self.action_dim, self.observation_dim))

    def train(self, episode=1, render=True):
        """
        Perform the training of the agents
        :param episode: number of game to make
        :param render: bool, Wether we want to display the game or not
        :return: None
        """
        last_train = 0
        last_reward = np.zeros(self.nb_agent)
        for _ in range(episode):
            state = self.env.reset()
            for d in range(self.horizon):
                if render:
                    self.env.render()

                actions = []
                for i in range(self.nb_agent):
                    actions.append(self.agents[i].explore(state[i]))

                next_state, rewards, done, info = self.env.step(actions)

                rewards = np.array(rewards)

                self.buffer.remember(state, actions, rewards, next_state)
                state = next_state

                if len(self.buffer.memory) < self.buffer.batch_size * 2 or last_train < 100:
                    last_train += 1
                    continue

                last_train = 0
                for i in range(self.nb_agent):
                    sample = self.buffer.sample()
                    self.train_step(sample, i)
                    self.agents[i].exploration_rate *= self.agents[i].exploration_decay

                self.update_targets()

    def train_step(self, sample, i):
        """
        Perform a training step of agent i on the sample
        :param sample: sample of transitions to train the agent on
        :param i: The id of the agent to train
        :return: None
        """
        y = []

        states = [[sample[j][0][l] for j in range(len(sample))] for l in range(self.nb_agent)]
        actions = [[sample[j][1][l] for j in range(len(sample))] for l in range(self.nb_agent)]

        s_in = {self.agents[i].observations_inputs[k]: states[k] for k in range(self.nb_agent)}
        a_in = {self.agents[i].actions_inputs[k]: actions[k]
                for k in range(self.nb_agent)}

        s_in_t = {self.agents[i].observations_inputs[k].name.split(":")[0]: np.asarray(states[k]) for k in range(self.nb_agent)}
        a_in_t = {self.agents[i].actions_inputs[k].name.split(":")[0]: np.array(actions[k])
                for k in range(self.nb_agent)}

        for state, actions, rewards, next_state in sample:

            # First we build the target value y:
            actionsp = []

            for k in range(self.nb_agent):
                agent = self.agents[k]
                observation = agent.watch(state, k)
                action = agent.target_action(observation)

                actionsp.append(action)

            yj = rewards[i] + self.gamma * self.agents[i].Q(next_state, actionsp, "target")
            y.append(yj)

        self.agents[i].critic.fit({**s_in_t, **a_in_t}, y)

        _ = self.session.run([self.agents[i].optimize_policy], feed_dict={**s_in, **a_in})

    def update_targets(self):
        """
        Make a training step of the target model according to target = tau * e + (1-tau) * target
        :return:
        """
        for i in range(self.nb_agent):
            self.session.run(self.agents[i].update_policy_target + self.agents[i].update_critic_target)
