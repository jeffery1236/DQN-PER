import numpy as np


class SumTree():
    ''' 
    A binary tree data structure where parent's value
    is the sum of its children
    '''

    def __init__(self, depth):
        self.depth = depth
        self.tree = np.zeros(2 * depth - 1)
        self.data = np.zeros(depth, dtype=object)
        self.n_entries = 0
        self.write = 0
    
    def _propagate(self, idx, change):
        # Update changes to root
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent > 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, val):
        left_child = 2 * idx + 1
        right_child = left_child + 1

        if left_child >= len(self.tree):
            return idx
        
        if val <= self.tree[left_child]:
            return self._retrieve(left_child, val)
        else:
            return self._retrieve(right_child, val - self.tree[left_child])
    
    def total(self):
        return self.tree[0]
    
# store priority and sample
    def add(self, p, data):
        idx = self.write + self.depth - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.depth:
            self.write = 0

        if self.n_entries < self.depth:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.depth + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    
