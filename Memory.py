from collections import namedtuple, deque
import torch
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, capacity, batch_size, n_env=1):
        self.keys = []
        self.n_env = n_env
        self.memory = {}
        self.index = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.transition = None

    def __len__(self):
        return self.index * self.n_env
    
    def memory_init(self, key, shape):
        steps_per_env = self.capacity // self.n_env
        shape = (steps_per_env, self.n_env,) + shape
        self.memory[key] = torch.zeros(shape)

    def add(self, **kwargs):
        if len(self.memory) == 0:
            for key in kwargs:
                self.keys.append(key)
                self.memory_init(key, tuple(kwargs[key].shape[1:]))
            self.transition = namedtuple('transition', self.keys)

        for key in kwargs:
            self.memory[key][self.index] = kwargs[key]

        self.index += 1

    def indices(self):
        ind = None
        if len(self) == self.capacity:
            ind = range(0, self.capacity)
        return ind

    def sample(self, indices, reshape_to_batch=True):
        if reshape_to_batch:
            values = [self.memory[k].reshape(-1, *self.memory[k].shape[2:]) for k in self.keys]
            result = self.transition(*values)
        else:
            values = [self.memory[k] for k in self.keys]
            result = self.transition(*values)

        return result

    def sample_batches(self, indices, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        values = [self.memory[k].reshape(-1, batch_size, *self.memory[k].shape[2:]) for k in self.keys]
        batch = self.transition(*values)
        return batch, self.capacity // batch_size

    def clear(self):
        self.index = 0