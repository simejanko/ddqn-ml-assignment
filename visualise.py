from matplotlib import pyplot as plt
import numpy as np


LOG_FILE = 'log.txt'
#only odd numbers
N_AVG = 1

data = np.loadtxt(LOG_FILE, skiprows=1)


if N_AVG == 1:
    frames = data[:, 0]
    average_reward = data[:, 1]
    average_Q = data[:, 2]
else:
    frames = data[int(N_AVG / 2):int(-N_AVG / 2), 0]
    average_reward = np.convolve(data[:, 1], np.ones((N_AVG,))/N_AVG, mode='valid')
    average_Q = np.convolve(data[:, 2], np.ones((N_AVG,))/N_AVG, mode='valid')

plt.plot(frames, average_reward)
plt.show()
plt.plot(frames, average_Q)
plt.show()