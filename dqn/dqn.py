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

#model that was used by Deepmind
DEEPMIND_MODEL = Sequential([
    Convolution2D(32, 8, 8, input_shape=(4,84,84), subsample=(4,4), activation='relu', dim_ordering='th'),
    Convolution2D(64, 4, 4, subsample=(2,2), activation='relu', dim_ordering='th'),
    Convolution2D(64, 3, 3, subsample=(1,1), activation='relu', dim_ordering='th'),
    Flatten(),
    Dense(512, activation="relu"),
])

class DDQN():
    @staticmethod
    def load(name, only_model = False):
        model = keras.models.load_model('{}.h5'.format(name))
        if only_model:
            dqn = DDQN(model, use_target=False)
        else:
            with open('{}.pkl'.format(name), 'rb') as file:
                dqn = pickle.load(file)
                dqn.replay_memory = ReplayMemory.load_by_chunks(file)

            dqn.model = model
            dqn.target_model = keras.models.load_model('{}_target.h5'.format(name))

        return dqn

    def __init__(self, model=None, n_actions=-1, use_target=True, replay_size=1000000, s_epsilon=1.0, e_epsilon=0.1,
                 f_epsilon=1000000, batch_size=32, gamma=0.99, hard_learn_interval=10000, warmup=50000,
                 priority_epsilon=0.02, priority_alpha=0.6, window_size = 4):
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
        :param window_size: Number of last observations to use as a single observation (accounting for transitions).
        """

        if model is None:
            #use default model
            model = DEEPMIND_MODEL
            model.add(Dense(n_actions, activation="linear"))
            model.compile(optimizer=Adam(lr=0.00025), loss='mse', metrics=[mean_squared_error]) #RMSprop(lr=0.00025)

        self.model = model
        if use_target:
            self.target_model = copy.deepcopy(model)
        else:
            self.target_model = model
        self.n_actions = model.layers[-1].output_shape[1]
        self.replay_memory = ReplayMemory(replay_size, window_size=window_size)
        self.epsilon = s_epsilon
        self.e_epsilon = e_epsilon
        self.d_epsilon = (e_epsilon - s_epsilon) / f_epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.hard_learn_interval = hard_learn_interval
        self.warmup = warmup
        self.priority_epsilon = priority_epsilon
        self.priority_alpha = priority_alpha
        self.window_size = window_size
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
        priority = math.fabs(t[a] - self._get_target(t[a], r, a_n, q_n, d))
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
        :param observation: Numpy array with the same shape as input Keras layer or
                            utils.ObservationSequenceStore object.
        :param use_epsilon: Enables/disables epsilon policy.
        """

        if use_epsilon and (random.random() < self.epsilon or self.step <= self.warmup):
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
            priority = self._get_propotional_priority(math.fabs(reward))
            self.replay_memory.add(priority, action, reward, new_observation, done)

        else:
            if self.epsilon > self.e_epsilon:
                self.epsilon += self.d_epsilon

            sample = self.replay_memory.sample(self.batch_size)
            idxs, prior_priorities, experiences = zip(*sample)
            self.replay_memory.add(3 * max(prior_priorities), action, reward, new_observation, done)

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

class GymDDQN(DDQN):
    """
    DDQN subclass that makes things easier to work with Gym environments
    (Only for envs with pixel state decriptors - eg. Atari)
    """

    @staticmethod
    def load(name, only_model = False, env_name="", actions_dict=None):
        if only_model:
            model = keras.models.load_model('{}.h5'.format(name))
            dqn = GymDDQN(env_name, model=model, only_model=True, actions_dict=actions_dict)
            return dqn

        dqn = DDQN.load(name)
        #TODO: remove bottom line with new run!!!
        dqn.only_model = False
        return dqn

    def __init__(self,env_name, actions_dict=None, cut_u=35, cut_d=15, h=84, only_model=False, **kwargs):
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
        self.only_model = only_model

        #TODO
        def _step(a):
            reward = 0
            action = self.env._action_set[a]
            lives_before = self.env.ale.lives()
            for _ in range(4):
                reward += self.env.ale.act(action)
            ob = self.env._get_obs()
            done = self.env.ale.game_over() or lives_before != self.env.ale.lives() or lives_before==0 and reward != 0
            return ob, reward, done, {}
        self.env._step = _step

        n_actions = self.env.action_space.n
        self.actions_dict = actions_dict
        if actions_dict is not None:
            n_actions = len(self.actions_dict)
        kwargs['n_actions'] = n_actions
        kwargs['use_target'] = not only_model
        super(GymDDQN, self).__init__(**kwargs)
        self._reset_episode()

    def _reset_episode(self):
        self.replay_memory.add(0, 0, 0, self._preprocess_observation(self.env.reset()), False)
        for i in range(self.window_size-1):
            o = self._preprocess_observation(self.env.step(self.env.action_space.sample())[0])

    def _preprocess_observation(self, o):
        return utils.preprocess_input(o, cut_u=self.cut_u, cut_d=self.cut_d, h=self.h)

    def learning_step(self):
        action, q_value = self.predict(self.replay_memory.get_last_observation(), use_epsilon=not self.only_model)
        gym_action = action
        if self.actions_dict is not None:
            gym_action = self.actions_dict[action]

        o, reward, done, _ = self.env.step(gym_action)
        o = self._preprocess_observation(o)

        if not self.only_model:
            super(GymDDQN, self).learning_step(action, reward, o, done)

        if done:
            self._reset_episode()
        return action, reward, q_value, done
