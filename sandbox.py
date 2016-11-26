import threading
import gym
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error
import gym_ple
from dqn.dqn import DQN
from curtsies import Input
import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np

render = False
def wait_input():
    global render
    with Input(keynames='curses') as input_generator:
        for e in input_generator:
            if e == 'r':
                render = not render

input_t = threading.Thread(target=wait_input)
input_t.start()

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def preprocess_input(image, cut_u, cut_d, h):
    image = rgb2gray(image[cut_u:-cut_d, :, :]).astype(int)
    aspect = image.shape[1]/image.shape[0]
    image = imresize(image, (h, int(h*aspect)), interp='bilinear')
    #plt.imshow(image, cmap='gray')
    #plt.show()
    return image

#TODO: every atari game has 6 action_space. Maybe try filtering. For example pong 3 possible actions: wait,up,down.
"""env = gym.make('Pong-v0')
observation = env.reset()
for i in range(95000):
    preprocess_input(observation, 35,15, 84)
    a = input()
    if not a:
        a = env.action_space.sample()
    else:
        a = int(a)
    observation, reward, done, _ = env.step(a)
    print("reward: %d" % reward)"""


"""model = Sequential()
model.add(Dense(20, input_shape=env.observation_space.shape , W_regularizer=l2(0.001)))
model.add(Activation("relu"))
model.add(Dense(8, W_regularizer=l2(0.001)))
model.add(Activation("relu"))
model.add(Dense(env.action_space.n, W_regularizer=l2(0.001)))
model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=[mean_squared_error])"""

#subsample=stride
env = gym.make('Pong-v0')
model = Sequential()
model.add(Convolution2D(16, 8, 8, input_shape=(2,84,84), subsample=(4,4), border_mode='valid', activation='relu', W_regularizer=l2(0.001)))
model.add(Convolution2D(32, 4, 4, subsample=(2,2), border_mode='valid', activation='relu', W_regularizer=l2(0.001)))
model.add(Convolution2D(32, 3, 3, subsample=(1,1), border_mode='valid', activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(128, W_regularizer=l2(0.001)))
model.add(Dense(env.action_space.n, W_regularizer=l2(0.001)))
model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=[mean_squared_error])

dqn = DQN(model, replay_size=50000, f_epsilon=500000, gamma=0.95)

#preprocess_input(observation, 35,15, 84)
for i_episode in range(500000):
    print(dqn.epsilon)
    observation = env.reset()
    done = False
    r_sum = 0
    while not done:
        if render:
            env.render()
        action = dqn.predict(observation)
        new_observation, reward, done, _ = env.step(action)
        #dqn.learning_step(observation, action, reward, new_observation, done)
        observation = new_observation
        r_sum += reward
    print("Episode {} finished with {} reward".format(i_episode, r_sum))
