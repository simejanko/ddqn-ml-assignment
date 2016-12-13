import numpy as np
import random
import copy
import pickle
import keras
from dqn.sum_tree import SumTree
import math
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error

#model that was used by Deepmind with 2 consecutive frames instead of 4
DEEPMIND_MODEL = Sequential([
    Convolution2D(32, 8, 8, input_shape=(2,84,84), subsample=(4,4), border_mode='valid', activation='relu', dim_ordering='th'),
    Convolution2D(64, 4, 4, subsample=(2,2), border_mode='valid', activation='relu', dim_ordering='th'),
    Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', activation='relu', dim_ordering='th'),
    Flatten(),
    Dense(512, activation="relu"),
])

#TODO: refactor only_model
class DDQN():
    @staticmethod
    def load(name, only_model = False):
        model = keras.models.load_model('{}.h5'.format(name))
        if only_model:
            dqn = DDQN(model, use_target=False)
        else:
            with open('{}.pkl'.format(name), 'rb') as file:
                dqn = pickle.load(file)
                dqn.replay_memory = SumTree.load_by_chunks(file)

            dqn.model = model
            dqn.target_model = keras.models.load_model('{}_target.h5'.format(name))

        return dqn

    def __init__(self, model=None, n_actions=-1, use_target=True, replay_size=100000, s_epsilon=1.0, e_epsilon=0.1,
                 f_epsilon=100000, batch_size=32, gamma=0.99, hard_learn_interval=10000, warmup=50000,
                 priority_epsilon=0.01, priority_alpha=0.6):
        """
        :param model: Keras neural network model.
        :param n_actions: Number of possible actions. Only used if using default model.
        :param use_target: Use target model or not.
        :param replay_size: Size of experience replay memory.
        :param s_epsilon: Start epsilon for Q-learning.
        :param e_epsilon: End epsilon for Q-learning.
        :param f_epsilon: Number of frames before epsilon gradually reaches e_epsilon.
        :param batch_size: Number of sampled experiences per frame.
        :param gamma: Future discount for Q-learning.
        :param hard_learn_interval: How often the target network is updated.
        :param warmup: Only perform random actions without learning for warmup steps.
        :param priority_epsilon: Added to every priority to avoid zero-valued priorities.
        :param priority_alpha: Between 0-1. Strength of priority experience sampling. 0 means uniform.
        """

        if model is None:
            #use default model
            model = DEEPMIND_MODEL
            model.add(Dense(n_actions, activation="linear"))
            model.compile(optimizer=RMSprop(lr=0.00025), loss='mse', metrics=[mean_squared_error])

        self.model = model
        if use_target:
            self.target_model = copy.deepcopy(model)
        else:
            self.target_model = model
        self.n_actions = model.layers[-1].output_shape[1]
        self.replay_memory = SumTree(replay_size)
        self.epsilon = s_epsilon
        self.e_epsilon = e_epsilon
        self.d_epsilon = (e_epsilon - s_epsilon) / f_epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.hard_learn_interval = hard_learn_interval
        self.warmup = warmup
        self.priority_epsilon = priority_epsilon
        self.priority_alpha = priority_alpha
        self.step = 1

    def _get_target(self, r, a_n, q_n, d):
        """
        Calculates double Q-learning target.
        """
        t = r
        if not d:
            t += self.gamma * q_n[a_n]
        return t

    def _modify_target(self, t, a, r, d, a_n, q_n):
        """
        Modifies target vector with DDQN target.
        """
        t[a] = self._get_target(r, a_n, q_n, d)
        return t

    def _get_propotional_priority(self, priority):
        return (priority + self.priority_epsilon)**self.priority_alpha

    def _get_priority(self, t, a, r, d, a_n, q_n):
        priority = math.fabs(t[a] - self._get_target(r, a_n, q_n, d))
        return self._get_propotional_priority(priority)


    def save(self, name, only_model=False):
        #it isn't recommended to pickle keras models. We don't pickle replay memory because of memory issues.
        self.model.save("{}.h5".format(name))

        if not only_model:
            self.target_model.save("{}_target.h5".format(name))

            model_tmp = self.model
            target_model_tmp = self.target_model
            replay_memory_tmp = self.replay_memory

            self.model = None
            self.target_model = None
            self.replay_memory = None
            with open("{}.pkl".format(name), "wb") as file:
                pickle.dump(self, file)
                replay_memory_tmp.save_by_chunks(file)

            self.model = model_tmp
            self.target_model = target_model_tmp
            self.replay_memory = replay_memory_tmp


    def predict(self, observation, use_epsilon=True):
        """
        Predicts next action with epsilon policy, given environment observation.
        :param observation: Numpy array with the same shape as input Keras layer.
        :param use_epsilon: Enables/disables epsilon policy.
        """

        if use_epsilon and (random.random() < self.epsilon or self.step <= self.warmup):
            return random.randint(0,self.n_actions-1)

        Q = self.model.predict_on_batch(np.expand_dims(observation, axis=0))[0]
        return np.argmax(Q)

    #TODO: Refactor. Prioritized experience replay kinda ruined it.
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
        if self.step <= self.warmup:
            #we use reward as priority during warmup
            priority = self._get_propotional_priority(math.fabs(reward))
            self.replay_memory.add(priority, (observation, action, reward, new_observation, done))

        else:
            if self.epsilon > self.e_epsilon:
                self.epsilon += self.d_epsilon

            sample = self.replay_memory.sample(self.batch_size)
            idxs, _, experiences = zip(*sample)
            #We need to do a forward pass on latest_experience to gets it's priority.
            #It is not used for learning though.
            last_experience = (observation, action, reward, new_observation, done)
            experiences += (last_experience, )

            obs, actions, rewards, obs2, dones = map(np.array, zip(*experiences))
            targets = self.model.predict_on_batch(obs)
            #double q learning target
            a_next = np.argmax(self.model.predict_on_batch(obs2), axis=1)
            Q_next = self.target_model.predict_on_batch(obs2)

            #calculate new priorities
            priorities = [self._get_priority(t, actions[i], rewards[i], dones[i], a_next[i], Q_next[i])
                          for i, t in enumerate(targets)]
            #update priorities and add latest experience to memory
            for idx, priority in zip(idxs, priorities[:-1]):
                self.replay_memory.update(idx, priority)
            self.replay_memory.add(priorities[-1], last_experience)

            #calculate new targets
            targets = np.array(
                [self._modify_target(t, actions[i], rewards[i], dones[i], a_next[i], Q_next[i])
                for i, t in enumerate(targets)])
            #latest experience is excluded from training
            self.model.train_on_batch(obs[:-1], targets[:-1])

            #Update target network - aka hard learning step
            if self.step % self.hard_learn_interval == 0:
                self.target_model.set_weights(self.model.get_weights())

        self.step += 1