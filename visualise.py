from matplotlib import pyplot as plt
import numpy as np
import json

#only odd numbers
N_AVG = 501
LOG_FILE = 'log.txt'
#COMPARISON_LOG_FILE = 'dqn_Breakout-v0_log.json'

data = np.loadtxt(LOG_FILE, skiprows=1)
#with open(COMPARISON_LOG_FILE) as data_file:
#    comparison_data = json.load(data_file)


if N_AVG == 1:
    frames = data[:, 0]
    average_reward = data[:, 1]
    average_Q = data[:, 2]
else:
    frames = data[int(N_AVG / 2):int(-N_AVG / 2), 0]
    average_reward = np.convolve(data[:, 1], np.ones((N_AVG,))/N_AVG, mode='valid')
    average_Q = np.convolve(data[:, 2], np.ones((N_AVG,))/N_AVG, mode='valid')

#frames2 = comparison_data['nb_steps'][int(N_AVG / 2):int(-N_AVG / 2)]
#average_reward2 = np.convolve(comparison_data['episode_reward'], np.ones((N_AVG,))/N_AVG, mode='valid')
#average_Q2 = np.convolve(comparison_data['mean_q'], np.ones((N_AVG,))/N_AVG, mode='valid')

plt.plot(frames, average_reward)
#plt.plot(frames2, average_reward2)
plt.show()
plt.plot(frames, average_Q)
#plt.plot(frames2, average_Q2)
plt.show()



