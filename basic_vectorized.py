import gym
from colorama.initialise import init
from keras.models import  Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error
import numpy as np
import random
import readchar
import threading

class ReplayMemory():
    def __init__(self, size):
        self.size = size
        self.memory = []

    def insert(self, s, a, r, s2, d):
        self.memory.append((s,a,r,s2,d))
        if len(self.memory) > self.size:
            del self.memory[0]

    def sample(self, n):
        if len(self.memory)<self.size:
            return []
        return random.sample(self.memory, n)

env = gym.make('LunarLander-v2')
model = Sequential()
model.add(Dense(12, input_shape=env.observation_space.shape , W_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dense(6, input_shape=env.observation_space.shape , W_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dense(env.action_space.n, W_regularizer=l2(0.01)))

#no activation at the end- linear
model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=[mean_squared_error])
replay_memory = ReplayMemory(25000)
EPSILON = 1
BATCH_SIZE = 32
GAMMA = 0.99
render = False

def wait_input():
    global render
    while True:
        c = readchar.readkey()
        if c == 'r':
            render = not render
        elif c == readchar.key.CTRL_C:
            break

def modify_target(t, a, r, d, n_m):
    t[a] = r
    if not d:
        t[a] += GAMMA * n_m
    return t

input_t = threading.Thread(target=wait_input)
input_t.start()

for i_episode in range(500000):
    print(EPSILON)
    observation = env.reset()
    done = False
    t_s = 0
    while not done:
        if render:
            env.render()
        #if t_s > 20000:
        #    break
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            Q = model.predict_on_batch(observation.reshape(1,-1))[0]
            action = np.argmax(Q)
        new_observation, reward, done, info = env.step(action)
        #reward = reward * t_s
        #if done:
        #    reward = -1
        replay_memory.insert(observation, action, reward, new_observation, done)
        experiences = replay_memory.sample(BATCH_SIZE)

        #TODO to mora bit vbistvu 2d array, ker rabmo targete tut za ostale akcije -> nastavs samo enako kot so bli napovedani
        #TODO podpri za poljubne dimenzije
        if(len(experiences) > 0):
            if EPSILON > 0.1:
                EPSILON -= 0.000005
            obs, actions, rewards, obs2, dones = map(np.array, zip(*experiences))
            targets = model.predict_on_batch(obs)
            next_max = np.max(model.predict_on_batch(obs2), axis=1)
            targets = np.array([modify_target(t, actions[i], rewards[i], dones[i], next_max[i]) for i,t in enumerate(targets)]) #TODO: also vectorize this if u can

            model.train_on_batch(obs, targets)

        t_s += 1
        observation = new_observation
    print("Episode {} finished after {} timesteps".format(i_episode, t_s+1))
