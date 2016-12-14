from matplotlib import pyplot as plt
import numpy as np


LOG_FILE = 'log.txt'
#only odd numbers
N_AVG = 21

#in the beggining we were already logging running average of 50...
#DIRTY_LOG_N = 450
DIRTY_LOG_N = 0


data = np.loadtxt(LOG_FILE, skiprows=1)
frames = data[int(N_AVG/2):int(-N_AVG/2), 0]
average_reward = np.append(data[:DIRTY_LOG_N,1],
                           np.convolve(data[DIRTY_LOG_N:, 1], np.ones((N_AVG,))/N_AVG, mode='valid'))


plt.plot(frames, average_reward)
plt.show()