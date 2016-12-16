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
    #dirty solution to prevent double saving
    just_loaded = True
else:
    with open("log.txt","w") as file:
        file.write("steps\treward\taverage action Q\n")

    dqn = DDQN(n_actions=env.action_space.n, replay_size=300000, f_epsilon=500000, gamma=0.99, warmup=100000)
    i_episode = 1
    just_loaded = False

log_batch=""
#preprocess_input(observation, 35,15, 84)
while i_episode < 50000000:
    if i_episode % 25 == 0 and not just_loaded:
        dqn.save("dqn_model")
        dqn.save("%s/dqn_model_%d" % (MODELS_DIR, i_episode), only_model=True)

        with open("log.txt", "a") as log:
            log.write(log_batch)
        log_batch = ""

    o1 = env.reset()
    o2 = env.step(env.action_space.sample())[0]
    o = utils.preprocess_input((o1, o2), 35, 15, 84)
    done = False
    #For logging reward and average action Q values
    r_sum = 0
    #q_values = []
    while not done:
        if render:
            env.render()
        #action, q_value = dqn.predict(o)
        #q_values.append(q_value)
        action = dqn.predict(o)
        o3, reward, done, _ = env.step(action)
        o_n = utils.preprocess_input((o2, o3), 35, 15, 84)
        dqn.learning_step(o, action, reward, o_n, done)
        o2 = o3
        o = o_n
        r_sum += reward

    print("Episode {} ({} steps) finished with {} reward".format(i_episode, dqn.step, r_sum))
    print("Epsilon:%f" % dqn.epsilon)

    #log_batch += "%d\t%f\t%f\n" % (dqn.step, r_sum, sum(q_values)/len(q_values))
    log_batch += "%d\t%f\n" % (dqn.step, r_sum)

    i_episode += 1
    just_loaded = False





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