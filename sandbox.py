import threading
import gym
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error
import gym_ple
from dqn.dqn import DDQN
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

def preprocess_input(images , cut_u, cut_d, h):
    output = []
    for image in images:
        image = rgb2gray(image[cut_u:-cut_d, :, :]).astype(int)
        aspect = image.shape[1]/image.shape[0]
        image = imresize(image, (h, int(h*aspect)), interp='bilinear')
        output.append(image)
    return np.array(output)
    #plt.imshow(image, cmap='gray')
    #plt.show()

#subsample=stride
#dim_ordering='th' - zato da je depth 0. dimenzija
#TODO: try Dropout
#TODO: every atari game has 6 action_space. Maybe try filtering. For example pong 3 possible actions: wait,up,down.
#TODO: refactor sandbox
open("log.txt","w").close()
env = gym.make('Pong-v0')
model = Sequential()
model.add(Convolution2D(16, 8, 8, input_shape=(2,84,84), subsample=(4,4), border_mode='valid', activation='relu', W_regularizer=l2(0.0001), dim_ordering='th'))
model.add(Convolution2D(32, 4, 4, subsample=(2,2), border_mode='valid', activation='relu', W_regularizer=l2(0.0001), dim_ordering='th'))
model.add(Convolution2D(32, 3, 3, subsample=(1,1), border_mode='valid', activation='relu', W_regularizer=l2(0.0001), dim_ordering='th'))
model.add(Flatten())
model.add(Dense(64, W_regularizer=l2(0.0001), activation="relu"))
model.add(Dense(env.action_space.n, W_regularizer=l2(0.0001), activation="linear"))
model.compile(optimizer=RMSprop(lr=0.0003), loss='mse', metrics=[mean_squared_error])

dqn = DDQN(model, replay_size=300000, f_epsilon=500000, gamma=0.99, warmup=100000)

r_sums = []
#preprocess_input(observation, 35,15, 84)
for i_episode in range(5000000):
    if len(r_sums)==50:
        with open("log.txt","a") as log:
            log.write("%f\n" % (sum(r_sums)/len(r_sums)))
            r_sums = []
            dqn.save("dqn_model")

    o1 = env.reset()
    o2 = env.step(env.action_space.sample())[0]
    o = preprocess_input((o1, o2), 35, 15, 84)
    done = False
    r_sum = 0
    while not done:
        if render:
            env.render()
        action = dqn.predict(o)
        o3, reward, done, _ = env.step(action)
        o_n = preprocess_input((o2, o3), 35, 15, 84)
        dqn.learning_step(o, action, reward, o_n, done)
        o2 = o3
        o = o_n
        r_sum += reward
    r_sums.append(r_sum)
    print("Episode {} ({} steps) finished with {} reward".format(i_episode, dqn.step, r_sum))
    print("Epsilon:%f" % dqn.epsilon)






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