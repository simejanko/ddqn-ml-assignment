import gym
from dqn.dqn import DDQN
import time
from dqn import utils
import os
from curtsies import Input
import threading

MODELS_DIR = 'models'

env = gym.make('Pong-v0')
dqns = []
for file in os.listdir(MODELS_DIR):
    f_no_ext = os.path.splitext(file)[0]
    dqn = DDQN.load('%s/%s' % (MODELS_DIR, f_no_ext), only_model=True)
    i_episode = int(f_no_ext.split("_")[-1])

    dqns.append((i_episode, dqn))

_, dqns = zip(*sorted(dqns))

i_model = 0
def wait_input():
    global i_model
    with Input(keynames='curses') as input_generator:
        for e in input_generator:
            if e == '+':
                i_model += 1
            elif e == '-':
                i_model -= 1


input_t = threading.Thread(target=wait_input)
input_t.start()

while True:
    o1 = env.reset()
    o2 = env.step(env.action_space.sample())[0]
    o = utils.preprocess_input((o1, o2), 35, 15, 84)
    done = False
    r_sum = 0
    while not done:
        env.render()
        action = dqns[i_model].predict(o, use_epsilon=False)
        o3, reward, done, _ = env.step(action)
        o_n = utils.preprocess_input((o2, o3), 35, 15, 84)
        o2 = o3
        o = o_n
        time.sleep(0.05)