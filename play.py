import gym
from dqn.dqn import DDQN
from scipy.misc import imresize
import numpy as np
import time

#TODO: implement these somewhere better (utils inside dqn package?)
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

env = gym.make('Pong-v0')
dqn = DDQN.load('models/dqn_model_450', only_model=True)

#preprocess_input(observation, 35,15, 84)
for i_episode in range(5000000):
    o1 = env.reset()
    o2 = env.step(env.action_space.sample())[0]
    o = preprocess_input((o1, o2), 35, 15, 84)
    done = False
    r_sum = 0
    while not done:
        env.render()
        action = dqn.predict(o, use_epsilon=False)
        o3, reward, done, _ = env.step(action)
        o_n = preprocess_input((o2, o3), 35, 15, 84)
        o2 = o3
        o = o_n
        time.sleep(0.05)