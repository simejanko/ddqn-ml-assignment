import numpy as np
import random
import copy
import pickle
import keras
from dqn.replay_memory import ReplayMemory
import math
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from keras.metrics import mean_squared_error
import dqn.utils as utils
import gym
import matplotlib.pyplot as plt
from keras import initializations

def weight_init(shape, name):
    return initializations.normal(shape, scale=0.04, name=name)

keras.initializations.weight_init = weight_init

#model that was used by Deepmind
DEEPMIND_MODEL = Sequential([
    Convolution2D(32, 8, 8, input_shape=(4,84,84), subsample=(4,4), init=weight_init, activation='relu', dim_ordering='th'),
    Convolution2D(64, 4, 4, subsample=(2,2), init=weight_init, activation='relu', dim_ordering='th'),
    Convolution2D(64, 3, 3, subsample=(1,1), init=weight_init, activation='relu', dim_ordering='th'),
    Flatten(),
    Dense(512, init=weight_init, activation="relu"),
])

class DDQN():
    @staticmethod
    def load(name, only_model = False):
        model = keras.models.load_model('{}.h5'.format(name))
        if only_model:
            dqn = DDQN(model, train=False)
        else:
            with open('{}.pkl'.format(name), 'rb') as file:
                dqn = pickle.load(file)
                dqn.replay_memory = ReplayMemory.load_by_chunks(file)

            dqn.model = model
            dqn.target_model = keras.models.load_model('{}_target.h5'.format(name))

        return dqn

    def __init__(self, model=None, n_actions=-1, train=True, replay_size=1000000, s_epsilon=1.0, e_epsilon=0.1,
                 f_epsilon=1000000, batch_size=32, gamma=0.99, hard_learn_interval=10000, warmup=50000,
                 priority_epsilon=0.02, priority_alpha=0.6, window_size = 4):
        """
        :param model: Keras neural network model.
        :param n_actions: Number of possible actions. Only used if using default model.
        :param train: Whether to train or not (test).
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
        :param window_size: Number of last observations to use as a single observation (accounting for transitions).
        """

        if model is None:
            #use default model
            model = DEEPMIND_MODEL
            model.add(Dense(n_actions, init=weight_init, activation="linear"))
            model.compile(optimizer=Adam(lr=0.00025), loss='mse')

        self.model = model
        self.n_actions = model.layers[-1].output_shape[1]
        self.replay_memory = ReplayMemory(replay_size, window_size=window_size)
        self.epsilon = s_epsilon
        self.e_epsilon = e_epsilon
        self.d_epsilon = (e_epsilon - s_epsilon) / f_epsilon
        self.batch_size = batch_size
        self.warmup = warmup
        self.gamma = gamma
        self.hard_learn_interval = hard_learn_interval
        self.priority_epsilon = priority_epsilon
        self.priority_alpha = priority_alpha
        self.window_size = window_size
        self.train = train

        if train:
            self.target_model = copy.deepcopy(model)
        else:
            self.target_model = model
            self.warmup = -1
            self.e_epsilon = s_epsilon

        self.step = 1

    def _get_target(self, orig, r, a_n, q_n, d):
        """
        Calculates double Q-learning target. Clips the diffrence from original value to [-1,1].
        """
        t = r
        if not d:
            t += self.gamma * q_n[a_n]

        #clipping
        if t-orig>1:
            t = orig +1
        elif t-orig<-1:
            t = orig -1
        return t

    def _modify_target(self, t, a, r, d, a_n, q_n):
        """
        Modifies target vector with DDQN target.
        """
        t[a] = self._get_target(t[a], r, a_n, q_n, d)
        return t

    def _get_propotional_priority(self, priority):
        return (priority + self.priority_epsilon)**self.priority_alpha

    def _get_priority(self, t, a, r, d, a_n, q_n):
        priority = abs(t[a] - self._get_target(t[a], r, a_n, q_n, d))
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


    def predict(self, observation):
        """
        Predicts next action with epsilon policy, given environment observation.
        :param observation: Numpy array with the same shape as input Keras layer or
                            utils.ObservationSequenceStore object.
        """

        if random.random() < self.epsilon or self.step <= self.warmup:
            return random.randint(0,self.n_actions-1), None

        Q = self.model.predict_on_batch(np.expand_dims(observation, axis=0))[0]
        a = np.argmax(Q)
        return a, Q[a]

    def learning_step(self, action, reward, new_observation, done):
        """
        Performs DDQN learning step
        :param action: Action performed.
        :param reward: Reward after performing the action.
        :param new_observation:Observation after performing the action.
        :param done: Bool - Is new state terminal.
        :return:
        """
        if self.step <= self.warmup:
            #we use reward as priority during warmup
            priority = self._get_propotional_priority(abs(reward))
            self.replay_memory.add(priority, action, reward, new_observation, done)

        else:
            if self.epsilon > self.e_epsilon:
                self.epsilon += self.d_epsilon

            sample = self.replay_memory.sample(self.batch_size)
            idxs, _, experiences = zip(*sample)

            obs, actions, rewards, obs2, dones = map(np.array, zip(*experiences))
            targets = self.model.predict_on_batch(obs)
            #double q learning target
            a_next = np.argmax(self.model.predict_on_batch(obs2), axis=1)
            Q_next = self.target_model.predict_on_batch(obs2)

            #calculate new priorities
            priorities = [self._get_priority(t, actions[i], rewards[i], dones[i], a_next[i], Q_next[i])
                          for i, t in enumerate(targets)]
            #update priorities and add latest experience to memory
            for idx, priority in zip(idxs, priorities):
                self.replay_memory.update(idx, priority)
            self.replay_memory.add(abs(3 * max(priorities)), action, reward, new_observation, done)
            #self.replay_memory.add(priorities, last_experience)

            #calculate new targets
            targets = np.array(
                [self._modify_target(t, actions[i], rewards[i], dones[i], a_next[i], Q_next[i])
                for i, t in enumerate(targets)])
            #latest experience is excluded from training
            self.model.train_on_batch(obs, targets)

            #Update target network - aka hard learning step
            if self.step % self.hard_learn_interval == 0:
                self.target_model.set_weights(self.model.get_weights())

        self.step += 1

class AtariDDQN(DDQN):
    """
    DDQN subclass that makes things easier to work with Atari environments
    (Only for envs with pixel state decriptors)
    """

    @staticmethod
    def load(name, only_model = False, env_name="", actions_dict=None):
        if only_model:
            model = keras.models.load_model('{}.h5'.format(name))
            dqn = AtariDDQN(env_name, model=model, actions_dict=actions_dict, train=False, s_epsilon=0.05)
            #dqn._set_step_func()
            return dqn

        dqn = DDQN.load(name)
        dqn._set_step_func()
        return dqn

    def __init__(self,env_name, actions_dict=None, cut_u=35, cut_d=15, h=84, **kwargs):
        """
        :param env_name: name of Gym environment.
        :param actions_dict: maps CNN output to Gym action.
         Only set this in case you don't want to use default Gym action space for that environment
         (Atari envs can for example contain some redundant actions). Note that this is still only used
         with default Deepmind model. In case any other model is given, user is expected to have correct
         number of output neurons for the given environment.
        :param observation_size: Number of consequtive frames to represent observation.
        :param cut_u: Observation preprocessing paramater. Look utils file for more info. Default: Atari parameters.
        :param cut_d: Observation preprocessing paramater. Look utils file for more info. Default: Atari parameters.
        :param h: Observation preprocessing paramater. Look utils file for more info. Default: Atari parameters.
        :param kwargs: arguments for DDQN class constructor.
        """

        self.env = gym.make(env_name)
        self.cut_u = cut_u
        self.cut_d = cut_d
        self.h = h

        n_actions = self.env.action_space.n
        self.actions_dict = actions_dict
        if actions_dict is not None:
            n_actions = len(self.actions_dict)
        kwargs['n_actions'] = n_actions
        super(AtariDDQN, self).__init__(**kwargs)
        if self.train:
            self._set_step_func()
        self._reset_episode()

    def _set_step_func(self):
        def _step(a):
            reward = 0
            action = self.env._action_set[a]
            lives_before = self.env.ale.lives()
            for _ in range(4):
                reward += self.env.ale.act(action)
            ob = self.env._get_obs()
            done = self.env.ale.game_over() or lives_before != self.env.ale.lives() or lives_before==0 and reward!=0
            return ob, reward, done, {}
        self.env._step = _step

    #mogoče probi non-zero majhne prioritiete
    def _reset_episode(self, hard_reset):
        n_frames = self.window_size
        if hard_reset:
            self.replay_memory.add(0, self.env.action_space.sample(), 0, self._preprocess_observation(self.env.reset()), False)
            n_frames -= 1
        for i in range(n_frames):
            a = self.env.action_space.sample()
            o = self._preprocess_observation(self.env.step(a)[0])
            self.replay_memory.add(0, a, 0 , o, False)

    def _preprocess_observation(self, o):
        return utils.preprocess_input(o, cut_u=self.cut_u, cut_d=self.cut_d, h=self.h)

    def learning_step(self):
        action, q_value = self.predict(np.array(self.replay_memory.get_last_observation()))
        gym_action = action
        if self.actions_dict is not None:
            gym_action = self.actions_dict[action]

        o, reward, done, _ = self.env.step(gym_action)
        o = self._preprocess_observation(o)

        if self.train:
            super(AtariDDQN, self).learning_step(action, reward, o, done)
        else:
            self.replay_memory.add(0, action, reward, o, done)

        game_over = self.env.ale.game_over()
        if done:
            self._reset_episode(game_over)
        return action, reward, q_value, game_over
