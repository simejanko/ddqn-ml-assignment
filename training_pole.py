import threading
import gym
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error
import gym_ple
from dqn.dqn import GymDDQN, DDQN
from dqn import utils
import numpy as np
import os
import copy


#0- neutral, 2-dup, 3-down
PONG_ACTIONS = [0,2,3]

MODELS_DIR = 'models'
OBSERVATION_SIZE = 4
render = False


env = gym.make('CartPole-v0')
model = Sequential()
model.add(Dense(15, input_shape=env.observation_space.shape , W_regularizer=l2(0.001)))
model.add(Activation("relu"))
model.add(Dense(8, W_regularizer=l2(0.001)))
model.add(Activation("relu"))
model.add(Dense(env.action_space.n, W_regularizer=l2(0.001)))
model.compile(optimizer=RMSprop(lr=0.0025), loss='mse', metrics=[mean_squared_error])
dqn = DDQN(model, replay_size=30000, f_epsilon=250000, gamma=1.0, hard_learn_interval=500, warmup=10000)

log = open("log.txt", "w")
log.write("steps\treward\taverage action Q\n")

i_episode = 1
while True:
    print(dqn.epsilon)
    o = env.reset()
    done = False
    r_sum = 0
    q_values = []
    while not done:
        if render:
            env.render()
        action, q_value = dqn.predict(o)
        if q_value is not None:
            q_values.append(q_value)
        o_n, reward, done, _ = env.step(action)
        if done:
            reward = -1
        dqn.learning_step(o, action, reward, o_n, done)
        o = o_n
        r_sum += reward
    q_avg = sum(q_values) / len(q_values) if len(q_values) > 0 else 0
    print("Episode {} finished with {} reward and avg. action q of {}".format(i_episode, r_sum, q_avg))
    log.write("{}\t{}\t{}\n".format(i_episode, r_sum,q_avg))

    i_episode += 1