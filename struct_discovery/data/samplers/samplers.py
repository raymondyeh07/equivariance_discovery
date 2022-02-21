"""Implements infinite looping dataloaders."""
import itertools
import math
import torch
from torch.utils.data.sampler import Sampler
import numpy as np


class TrainingSampler(Sampler):
    def __init__(self, size, shuffle=True, seed=None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        assert size > 0
        self._size = size
        self._shuffle = shuffle
        if seed is None:
            seed = np.random.randint(2 ** 31)
        self._seed = int(seed)

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


InferenceSampler = Sampler
