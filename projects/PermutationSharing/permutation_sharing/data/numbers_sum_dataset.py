"""Implements sum of numbers dataset."""

import numpy as np
import torch
from torch.utils import data


class NumbersSumDataset(data.Dataset):
    def __init__(self, num_samples, max_length=10, A=None, A_rank=1, w_scale=1,
                 add_noise=0, rand_seed=None):
        self.num_samples = num_samples
        self.max_length = max_length
        self.A_rank = A_rank
        self.w_scale = w_scale
        self.rand_seed = rand_seed

        if rand_seed is not None:
            np.random.seed(rand_seed)

        # Generated data.
        X = np.random.randint(1, 10, size=(num_samples, max_length))

        if A is None:
            A = np.zeros((max_length, max_length))
            for kk in range(A_rank):
                A[kk, kk] = 1
            for kk in range(A_rank, max_length):
                tmp = kk % A_rank
                A[kk, tmp] = 1
        else:
            assert A.shape[0] == max_length

        self.A = A
        if A_rank == 2:
            weight = [w_scale, 1]*((max_length+1)//2)
            weight = np.array(weight[:max_length])
        elif A_rank == 1:
            weight = np.ones(max_length)

        self.data = X
        self.label = np.sum(X*weight, 1, keepdims=True).astype(np.float32)
        if add_noise > 0:
            self.label += np.random.uniform(-add_noise,
                                            add_noise, size=(num_samples, 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.label[index]
