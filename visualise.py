from matplotlib import pyplot as plt
import numpy as np
import json
import matplotlib

matplotlib.rcParams.update({'font.size': 16})

#only odd numbers
N_AVG = 51
LOG_FILE = 'log.txt'

data = np.loadtxt(LOG_FILE, skiprows=1)


if N_AVG == 1:
    frames = data[:, 0]/1000000
    average_reward = data[:, 1]
    average_Q = data[:, 2]
else:
    frames = data[int(N_AVG / 2):int(-N_AVG / 2), 0]/1000000
    average_reward = np.convolve(data[:, 1], np.ones((N_AVG,))/N_AVG, mode='valid')
    average_Q = np.convolve(data[:, 2], np.ones((N_AVG,))/N_AVG, mode='valid')

plt.plot(frames, average_reward)
plt.xlabel("Number of training frames (in millions)")
plt.ylabel("Running average reward")
plt.show()
plt.plot(frames, average_Q)
plt.xlabel("Number of training frames (in millions)")
plt.ylabel("Average action Q-value")
plt.show()