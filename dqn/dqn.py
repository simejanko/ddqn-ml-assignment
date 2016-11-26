import numpy as np
import random


class ReplayMemory():
    """
    Experience replay memory.
    Stores (state, action, reward, new_state, is_terminal_state) tuples.
    """

    def __init__(self, size):
        self.size = size
        self.memory = []

    def insert(self, s, a, r, s2, d):
        self.memory.append((s,a,r,s2,d))
        if len(self.memory) > self.size:
            del self.memory[0]

    def sample(self, n):
        """
        :return: Memory sample of size n if full else empty.
        """
        if len(self.memory) < self.size:
            return []
        return random.sample(self.memory, n)

class DQN():
    def __init__(self, model, replay_size=10000, s_epsilon=1.0, e_epsilon=0.1,
                 f_epsilon=100000, batch_size=32, gamma=0.95):
        """
        :param model: Keras neural network model.
        :param replay_size: Size of experience replay memory.
        :param s_epsilon: Start epsilon for Q-learning.
        :param e_epsilon: End epsilon for Q-learning.
        :param f_epsilon: Number of frames before epsilon gradually reaches e_epsilon.
        :param batch_size: Number of sampled experiences per frame.
        :param gamma: Future discount for Q-learning.
        """

        self.model = model
        self.n_actions = model.layers[-1].output_shape[1]
        self.replay_memory = ReplayMemory(replay_size)
        self.epsilon = s_epsilon
        self.e_epsilon = e_epsilon
        self.d_epsilon = (e_epsilon - s_epsilon) / f_epsilon
        self.batch_size = batch_size
        self.gamma = gamma

    def _modify_target(self, t, a, r, d, n_m):
        t[a] = r
        if not d:
            t[a] += self.gamma * n_m
        return t

    def predict(self, observation):
        """
        Predicts next action with epsilon policy, given environment observation.
        :param observation: Numpy array with the same shape as input Keras layer.
        """

        if random.random() < self.epsilon:
            return random.randint(0,self.n_actions-1)

        Q = self.model.predict_on_batch(observation)[0]
        return np.argmax(Q)

    def learning_step(self, observation, action, reward, new_observation, done):
        """
        Performs DQN learning step
        :param observation: Observation before performing the action.
        :param action: Action performed.
        :param reward: Reward after performing the action.
        :param new_observation:Observation after performing the action.
        :param done: Bool - Is new state/observation terminal.
        :return:
        """

        self.replay_memory.insert(observation, action, reward, new_observation, done)
        experiences = self.replay_memory.sample(self.batch_size)

        if (len(experiences) > 0):
            if self.epsilon > self.e_epsilon:
                self.epsilon += self.d_epsilon

            obs, actions, rewards, obs2, dones = map(np.array, zip(*experiences))
            targets = self.model.predict_on_batch(obs)
            next_max = np.max(self.model.predict_on_batch(obs2), axis=1)
            targets = np.array(
                [self._modify_target(t, actions[i], rewards[i], dones[i], next_max[i]) for i, t in enumerate(targets)])

            self.model.train_on_batch(obs, targets)
        return observation
