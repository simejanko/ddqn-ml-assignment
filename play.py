import gym
from dqn.dqn import DDQN
import time
from dqn import utils


env = gym.make('Pong-v0')
dqn = DDQN.load('models/dqn_model_450', only_model=True)

#preprocess_input(observation, 35,15, 84)
for i_episode in range(5000000):
    o1 = env.reset()
    o2 = env.step(env.action_space.sample())[0]
    o = utils.preprocess_input((o1, o2), 35, 15, 84)
    done = False
    r_sum = 0
    while not done:
        env.render()
        action = dqn.predict(o, use_epsilon=False)
        o3, reward, done, _ = env.step(action)
        o_n = utils.preprocess_input((o2, o3), 35, 15, 84)
        o2 = o3
        o = o_n
        time.sleep(0.05)