#Modified version of: https://github.com/jaara/AI-blog/blob/master/SumTree.py
import numpy
import numpy as np
import pickle
import matplotlib.pyplot as plt
class SumTree:
    write = 0

    @staticmethod
    def load_by_chunks(file):
        """
        Loads SumTree that was pickled by chunks.
        :param file: File object.
        """
        sum_tree = pickle.load(file)
        sum_tree.data = pickle.load(file)
        #sum_tree.data = numpy.zeros( sum_tree.capacity, dtype=object )
        while True:
            try:
                sum_tree.data = np.append(sum_tree.data, pickle.load(file))
            except EOFError:
                break

        return sum_tree

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def save_by_chunks(self, file, chunk_size=2000):
        """
        Pickles SumTree by chunks. Slower but uses less memory.
        :param file: File object.
        :param chunk_size: Size of a single chunk.
        """
        data_temp = self.data
        self.data = None

        pickle.dump(self, file)
        self.data = data_temp

        for i in range(0, self.data.size, chunk_size):
            end_i = i+chunk_size
            if end_i > self.data.size:
                end_i = self.data.size

            pickle.dump(self.data[i:end_i], file)

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

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
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

        return (idx, self.tree[idx], self.data[dataIdx])

    def sample(self, n):
        s_vec = numpy.random.uniform(0.0, self.tree[0], n)
        return [self.get(s) for s in s_vec]

    def hist(self, **kwargs):
        plt.hist(self.tree[self.capacity:], **kwargs)
        plt.show()