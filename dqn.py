import gym
from colorama.initialise import init
from keras.models import  Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error
import gym_ple
import numpy as np
import random
import readchar
import threading

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

#TODO: decouple environment...
class DQN():
    def __init__(self, model, env_step, env_render=None, replay_size=10000, s_epsilon=1.0, e_epsilon=0.1,
                 f_epsilon=10000, batch_size=32, gamma=0.95):
        """
        :param model: Keras neural network model.
        :param env_step: Environment step function that accepts action and returns
               (new_state, reward, is_new_state_terminal, meta-info) tuple
        :param env_render: Environment render function (Optional).
        :param replay_size: Size of experience replay memory.
        :param s_epsilon: Start epsilon for Q-learning.
        :param e_epsilon: End epsilon for Q-learning.
        :param f_epsilon: Number of frames before epsilon gradually reaches e_epsilon.
        :param batch_size: Number of sampled experiences per frame.
        :param gamma: Future discount for Q-learning.
        """

        self.model = model
        self.n_actions = model.layers[-1].output_shape[1]
        self.env_step = env_step
        self.env_render = env_render
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

        Q = self.model.predict_on_batch(observation.reshape(1, -1))[0]
        return np.argmax(Q)

    def learning_step(self, observation, render=True):
        """
        Preform DQN learning step and return new observation.
        """
        if render and self.env_render != None:
            self.env_render()

        action = self.predict(observation)
        new_observation, reward, done, _ = self.env_step(action)
        self.replay_memory.insert(observation, action, reward, new_observation, done)
        experiences = self.replay_memory.sample(self.batch_size)

        if (len(experiences) > 0):
            if self.epsilon > self.e_epsilon:
                self.epsilon -= self.d_epsilon

            obs, actions, rewards, obs2, dones = map(np.array, zip(*experiences))
            targets = self.model.predict_on_batch(obs)
            next_max = np.max(self.model.predict_on_batch(obs2), axis=1)
            targets = np.array(
                [self._modify_target(t, actions[i], rewards[i], dones[i], next_max[i]) for i, t in enumerate(targets)])

            self.model.train_on_batch(obs, targets)
        return observation

render = False
def wait_input():
    global render
    global reset
    while True:
        c = readchar.readkey()
        if c=='s':
            reset = True
        elif c == readchar.key.CTRL_C:
            break

input_t = threading.Thread(target=wait_input)
input_t.start()

env = gym.make('CartPole-v0')
model = Sequential()
model.add(Dense(12, input_shape=env.observation_space.shape , W_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dense(6, W_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dense(env.action_space.n, W_regularizer=l2(0.01)))
model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=[mean_squared_error])

dqn = DQN(model, env.step, replay_size=25000)


for i_episode in range(500000):
    print(dqn.epsilon)
    observation = env.reset()
    done = False
    t_s = 0
    r_sum = 0
    while not done:
        if render:
            env.render()
        observation =dqn.learning_step(observation, render=False)

        t_s += 1
    print("Episode {} finished with {} reward".format(i_episode, t_s+1))
