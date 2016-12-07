import threading
import gym
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error
import gym_ple
from dqn.dqn import DDQN
from dqn import utils
import numpy as np
import os

MODELS_DIR = 'models'
render = False
"""def wait_input():
    global render
    with Input(keynames='curses') as input_generator:
        for e in input_generator:
            if e == 'r':
                render = not render


input_t = threading.Thread(target=wait_input)
input_t.start()"""


#TODO: try Dropout
#TODO: every atari game has 6 action_space. Maybe try filtering. For example pong 3 possible actions: wait,up,down.
#TODO: refactor sandbox
env = gym.make('Pong-v0')
if os.path.isfile('dqn_model.pkl'):
    dqn = DDQN.load('dqn_model')
    i_episode = max([int(os.path.splitext(file)[0].split("_")[-1]) for file in os.listdir(MODELS_DIR)])
else:
    open("log.txt","w").close()

    # subsample=stride
    # dim_ordering='th' - zato da je depth 0. dimenzija
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, input_shape=(2,84,84), subsample=(4,4), border_mode='valid', activation='relu', dim_ordering='th'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2), border_mode='valid', activation='relu', dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', activation='relu', dim_ordering='th'))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(env.action_space.n,  activation="linear"))
    model.compile(optimizer=RMSprop(lr=0.00025), loss='mse', metrics=[mean_squared_error])

    dqn = DDQN(model, replay_size=300000, f_epsilon=500000, gamma=0.995, warmup=100000)
    i_episode = 0

r_sums = []
#preprocess_input(observation, 35,15, 84)
while i_episode < 5000000:
    if len(r_sums)==50:
        with open("log.txt","a") as log:
            log.write("%f\n" % (sum(r_sums)/len(r_sums)))
            r_sums = []
            dqn.save("dqn_model")
            dqn.save("%s/dqn_model_%d" % (MODELS_DIR, i_episode), only_model=True)

    o1 = env.reset()
    o2 = env.step(env.action_space.sample())[0]
    o = utils.preprocess_input((o1, o2), 35, 15, 84)
    done = False
    r_sum = 0
    while not done:
        if render:
            env.render()
        action = dqn.predict(o)
        o3, reward, done, _ = env.step(action)
        o_n = utils.preprocess_input((o2, o3), 35, 15, 84)
        dqn.learning_step(o, action, reward, o_n, done)
        o2 = o3
        o = o_n
        r_sum += reward
    r_sums.append(r_sum)
    print("Episode {} ({} steps) finished with {} reward".format(i_episode, dqn.step, r_sum))
    print("Epsilon:%f" % dqn.epsilon)

    i_episode += 1






"""env = gym.make('CartPole-v0')
model = Sequential()
model.add(Dense(20, input_shape=env.observation_space.shape , W_regularizer=l2(0.001)))
model.add(Activation("relu"))
model.add(Dense(8, W_regularizer=l2(0.001)))
model.add(Activation("relu"))
model.add(Dense(env.action_space.n, W_regularizer=l2(0.001)))
model.compile(optimizer=RMSprop(lr=0.0025), loss='mse', metrics=[mean_squared_error])
dqn = DDQN(model, replay_size=30000, f_epsilon=250000, gamma=1.0, hard_learn_interval=500, warmup=10000)"""

"""for i_episode in range(5000000):
    print(dqn.epsilon)
    o = env.reset()
    done = False
    r_sum = 0
    while not done:
        if render:
            env.render()
        action = dqn.predict(o)
        o_n, reward, done, _ = env.step(action)
        if done:
            reward = -1
        dqn.learning_step(o, action, reward, o_n, done)
        o = o_n
        r_sum += reward
    print("Episode {} finished with {} reward".format(i_episode, r_sum))"""