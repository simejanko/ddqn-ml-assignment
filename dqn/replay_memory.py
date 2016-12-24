#Modified version of: https://github.com/jaara/AI-blog/blob/master/SumTree.py
import numpy
import numpy as np
import pickle
import matplotlib.pyplot as plt
class ReplayMemory:
    write = 0

    @staticmethod
    def load_by_chunks(file):
        """
        Loads SumTree that was pickled by chunks.
        :param file: File object.
        """
        sum_tree = pickle.load(file)
        sum_tree.frame_data = pickle.load(file)
        #sum_tree.data = numpy.zeros( sum_tree.capacity, dtype=object )
        while True:
            try:
                sum_tree.frame_data = np.append(sum_tree.frame_data, pickle.load(file))
            except EOFError:
                break

        return sum_tree

    def __init__(self, capacity, window_size=1):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )
        self.frame_data = numpy.zeros(capacity+window_size , dtype=object)
        self.window_size = window_size

    def save_by_chunks(self, file, chunk_size=2000):
        """
        Pickles SumTree by chunks. Slower but uses less memory.
        :param file: File object.
        :param chunk_size: Size of a single chunk.
        """
        frame_data_temp = self.frame_data
        self.frame_data = None

        pickle.dump(self, file)
        self.frame_data = frame_data_temp

        for i in range(0, self.frame_data.size, chunk_size):
            end_i = i+chunk_size
            if end_i > self.frame_data.size:
                end_i = self.frame_data.size

            pickle.dump(self.frame_data[i:end_i], file)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def add(self, p, a, r, obs2, d):
        idx = self.write + self.capacity - 1

        self.frame_data[self.write+self.window_size] = obs2
        if self.write == 0:
            self.frame_data[self.write: self.window_size] = [obs2] * self.window_size
        self.data[self.write] = (self.write, a, r, self.write+1, d)
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0


    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        obsIdx, a, r, obs2Idx, d = self.data[dataIdx]
        obs, obs2 = self.frame_data[obsIdx:obsIdx+self.window_size], self.frame_data[obs2Idx:obs2Idx+self.window_size]
        return idx, self.tree[idx], (obs, a, r, obs2, d)

    def get_last_observation(self):
        dataIdx = self.write-1
        obsIdx, a, r, obs2Idx, d = self.data[dataIdx]
        obs2 = self.frame_data[obs2Idx:obs2Idx + self.window_size]
        return obs2

    def sample(self, n):
        s_vec = numpy.random.uniform(0.0, self.tree[0], n)
        return [self.get(s) for s in s_vec]

    def hist(self, **kwargs):
        plt.hist(self.tree[self.capacity:], **kwargs)
        plt.show()