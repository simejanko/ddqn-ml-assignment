import gym
from dqn.dqn import GymDDQN
import time
from dqn import utils
import os
from curtsies import Input
import threading

#0- neutral, 2-dup, 3-down
PONG_ACTIONS = [0,2,3]
MODELS_DIR = 'models'

dqns = []
for file in os.listdir(MODELS_DIR):
    f_no_ext = os.path.splitext(file)[0]
    dqn = GymDDQN.load('%s/%s' % (MODELS_DIR, f_no_ext), only_model=True,  env_name='Breakout-v0')
    i_episode = int(f_no_ext.split("_")[-1])

    dqns.append((i_episode, dqn))

episodes, dqns = zip(*sorted(dqns))

i_model = 0
def wait_input():
    global i_model
    global episodes
    with Input(keynames='curses') as input_generator:
        for e in input_generator:
            if e == '+':
                i_model += 1
                print(episodes[i_model])
            elif e == '-':
                i_model -= 1
                print(episodes[i_model])


input_t = threading.Thread(target=wait_input)
input_t.start()

while True:
    done = False
    while not done:
        dqns[i_model].env.render()
        dqns[i_model].learning_step()
        time.sleep(0.05)