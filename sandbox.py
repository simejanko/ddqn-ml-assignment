import readchar
import threading
import gym
from keras.models import  Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error
import gym_ple
from dqn.dqn import DQN

render = False
def wait_input():
    global render
    while True:
        c = readchar.readkey()
        if c=='r':
            render = not render
        elif c == readchar.key.CTRL_C:
            break

input_t = threading.Thread(target=wait_input)
input_t.start()

env = gym.make('CartPole-v0')
model = Sequential()
model.add(Dense(12, input_shape=env.observation_space.shape , W_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dense(6, W_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dense(env.action_space.n, W_regularizer=l2(0.01)))
model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=[mean_squared_error])

dqn = DQN(model, replay_size=25000)


for i_episode in range(500000):
    print(dqn.epsilon)
    observation = env.reset()
    done = False
    t_s = 0
    r_sum = 0
    while not done:
        if render:
            env.render()
        action = dqn.predict(observation)
        new_observation, reward, done, _ = env.step(action)
        dqn.learning_step(observation, action, reward, new_observation, done)

        t_s += 1
    print("Episode {} finished with {} reward".format(i_episode, t_s+1))
