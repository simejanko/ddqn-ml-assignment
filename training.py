from dqn.dqn import AtariDDQN
import os
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error

#TODO: probi nearest namest bilinear pa probi brez prioritized experience replaya
#0- neutral, 2-dup, 3-down
PONG_ACTIONS = [0,2,3]
MODELS_DIR = 'models'
OBSERVATION_SIZE = 4
render = False

"""model = Sequential([
    Convolution2D(16, 8, 8, input_shape=(4,84,84), subsample=(4,4), border_mode='valid', activation='relu', dim_ordering='th'),
    Convolution2D(32, 4, 4, subsample=(2,2), border_mode='valid', activation='relu', dim_ordering='th'),
    Convolution2D(32, 3, 3, subsample=(1,1), border_mode='valid', activation='relu', dim_ordering='th'),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(6, activation="linear"),

])
model.compile(optimizer=RMSprop(lr=0.00025), loss='mse', metrics=[mean_squared_error])"""

if os.path.isfile('dqn_model.pkl'):
    dqn = AtariDDQN.load('dqn_model')
    i_episode = max([int(os.path.splitext(file)[0].split("_")[-1]) for file in os.listdir(MODELS_DIR)])
    #dirty solution to prevent double saving
    just_loaded = True
else:
    with open("log.txt","w") as file:
        file.write("steps\treward\taverage action Q\n")
    dqn = AtariDDQN('Pong-v0', replay_size=850000)
    i_episode = 1
    just_loaded = False

log_batch=""
while True:
    if i_episode % 25 == 0 and not just_loaded:
        dqn.save("dqn_model")
        dqn.save("%s/dqn_model_%d" % (MODELS_DIR, i_episode), only_model=True)

        with open("log.txt", "a") as log:
            log.write(log_batch)
        log_batch = ""

    done = False
    #For logging reward and average action Q values
    r_sum = 0
    q_values = []
    while not done:
        if render:
            dqn.env.render()
        #where the actual learning happens. Most of everything else is logging.
        action, reward, q_value, done = dqn.learning_step()
        r_sum += reward
        if q_value is not None:
            q_values.append(q_value)

    q_avg = sum(q_values)/len(q_values) if len(q_values) > 0 else 0
    log_batch += "%d\t%f\t%f\n" % (dqn.step, r_sum, q_avg)

    print("Episode {} ({} steps) finished with {} reward and {:.2f} avg. Q".format(i_episode, dqn.step, r_sum, q_avg))
    print("Epsilon:%f" % dqn.epsilon)

    i_episode += 1
    just_loaded = False