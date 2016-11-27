import numpy as np
import random
import copy
import pickle
import keras

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
        return random.sample(self.memory, n)

#TODO: test save/load
class DDQN():
    def __init__(self, model, replay_size=100000, s_epsilon=1.0, e_epsilon=0.1,
                 f_epsilon=100000, batch_size=32, gamma=0.99, hard_learn_interval=10000, warmup=50000):
        """
        :param model: Keras neural network model.
        :param replay_size: Size of experience replay memory.
        :param s_epsilon: Start epsilon for Q-learning.
        :param e_epsilon: End epsilon for Q-learning.
        :param f_epsilon: Number of frames before epsilon gradually reaches e_epsilon.
        :param batch_size: Number of sampled experiences per frame.
        :param gamma: Future discount for Q-learning.
        :param hard_learn_interval: How often the target network is updated.
        :param warmup: Only perform random actions without learning for warmup steps.
        """

        self.model = model
        self.target_model = copy.deepcopy(model)
        self.n_actions = model.layers[-1].output_shape[1]
        self.replay_memory = ReplayMemory(replay_size)
        self.epsilon = s_epsilon
        self.e_epsilon = e_epsilon
        self.d_epsilon = (e_epsilon - s_epsilon) / f_epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.hard_learn_interval = hard_learn_interval
        self.warmup = warmup
        self.step = 1

    def save(self, name):
        #it isn't recommended to pickle keras models
        self.model.save("{}.h5".format(name))
        self.target_model.save("{}_target.h5".format(name))

        odict = self.__dict__.copy()
        del odict['model']
        del odict['target_model']
        with open("{}.pkl".format(name), 'wb') as file:
            pickle.dump(odict, file)

    def load(self, name):
        with open('{}.pkl'.format(name), 'rb') as file:
            self.__dict__ = pickle.load(file)

        self.model = keras.models.load_model('{}.h5'.format(name))
        self.target_model = keras.models.load_model('{}_target.h5'.format(name))

    def _modify_target(self, t, a, r, d, a_n, q_n):
        """
        Sets double Q-learning target.
        """
        t[a] = r
        if not d:
            t[a] += self.gamma * q_n[a_n]
        return t

    def predict(self, observation, use_epsilon=True):
        """
        Predicts next action with epsilon policy, given environment observation.
        :param observation: Numpy array with the same shape as input Keras layer.
        :param use_epsilon: Enables/disables epsilon policy.
        """

        if use_epsilon and random.random() < self.epsilon or self.step <= self.warmup:
            return random.randint(0,self.n_actions-1)

        Q = self.model.predict_on_batch(np.expand_dims(observation, axis=0))[0]
        return np.argmax(Q)

    def learning_step(self, observation, action, reward, new_observation, done):
        """
        Performs DDQN learning step
        :param observation: Observation before performing the action.
        :param action: Action performed.
        :param reward: Reward after performing the action.
        :param new_observation:Observation after performing the action.
        :param done: Bool - Is new state terminal.
        :return:
        """

        self.replay_memory.insert(observation, action, reward, new_observation, done)

        if self.step > self.warmup:
            experiences = self.replay_memory.sample(self.batch_size)

            if self.epsilon > self.e_epsilon:
                self.epsilon += self.d_epsilon

            obs, actions, rewards, obs2, dones = map(np.array, zip(*experiences))
            targets = self.model.predict_on_batch(obs)
            #double q learning target
            a_next = np.argmax(self.model.predict_on_batch(obs2), axis=1)
            Q_next = self.target_model.predict_on_batch(obs2)
            targets = np.array(
                [self._modify_target(t, actions[i], rewards[i], dones[i], a_next[i], Q_next[i]) for i, t in enumerate(targets)])

            self.model.train_on_batch(obs, targets)

            #Update target network - aka hard learning step
            if self.step % self.hard_learn_interval == 0:
                self.target_model.set_weights(self.model.get_weights())

        self.step += 1
        return observation
