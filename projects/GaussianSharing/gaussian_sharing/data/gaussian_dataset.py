"""Implements Gaussian Dataset."""

import numpy as np

import torch
from torch.utils import data

from .helpers import partition


class GaussianDataset(data.Dataset):
    """Generate Gaussian dataset."""

    def __init__(self, kdim=1, minmax_cluster_num=[-1, -1], num_samples=10,
                 cluster_mu_diff=3, cluster_std=1,
                 rand_seed=None, A=None, data=None):
        self.kdim = kdim
        self.max_cluster_num = minmax_cluster_num
        max_cluster_num = minmax_cluster_num
        self.num_samples = num_samples
        self.cluster_mu_diff = cluster_mu_diff
        self.cluster_std = cluster_std
        self.rand_seed = rand_seed

        if rand_seed is not None:
            np.random.seed(rand_seed)

        if max_cluster_num[0] < 0:
            max_cluster_num[0] = 1
        if max_cluster_num[1] < 0:
            max_cluster_num[1] = kdim

        assert kdim >= max_cluster_num[1]

        # Genearte random A.
        if A is None:
            if minmax_cluster_num[0] == minmax_cluster_num[0] and kdim > 6:
                # Rank is fixed no sampling necessary.
                A = np.zeros((kdim, kdim))
                for kk in range(minmax_cluster_num[0]):
                    A[kk, kk] = 1
                for kk in range(minmax_cluster_num[0], self.kdim):
                    tmp = np.random.randint(0, minmax_cluster_num[0])
                    A[kk, tmp] = 1
            else:
                A = np.zeros((kdim, kdim))
                # Generate parition.
                group = list(partition(list(range(0, kdim))))
                group_sub = [cluster for cluster in group if (len(
                    cluster) >= minmax_cluster_num[0]) and (len(cluster) <= minmax_cluster_num[1])]
                sample_group_idx = np.random.randint(0, len(group_sub))
                sample_group = group_sub[sample_group_idx]
                for cnum, cluster in enumerate(sample_group):
                    A[cnum, cluster] = 1
                A = A.T
        else:
            group = np.argmax(A, -1)
        self.A = A
        psi = (np.arange(0, kdim))*cluster_mu_diff
        group = np.argmax(A, -1)
        theta = psi[group]
        self.theta_gt = theta
        # Generate data
        self.data = np.random.randn(
            num_samples, kdim)*cluster_std + self.theta_gt
        self.data = self.data.astype('float32')

        if data is not None:
            assert A is not None
            self.data = data
            self.num_samples = data.shape[0]

        assert(np.array_equal(self.A.dot(psi), theta))
        # Get ground-truth values
        self.theta_hat = np.mean(self.data, 0)
        self.l2_gt_loss = np.mean(np.sum((self.data-self.theta_gt)**2, -1))
        self.mse_gt_loss = np.sum((self.theta_hat-self.theta_gt)**2)
        self.l2_hat_loss = np.mean(np.sum((self.data-self.theta_hat)**2, -1))
        self.A_gt = self.A

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.data[index]
