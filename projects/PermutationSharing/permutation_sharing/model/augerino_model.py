"""Implements a structured fully connected layer."""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from struct_discovery.layers.struct_base_layer import StructBaseLayer


# Gumbel sinkhorn support from https://github.com/perrying/gumbel-sinkhorn/blob/master/utils/gumbel_sinkhorn_ops.py
def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int = 20) -> (torch.Tensor,):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0, n_iter: int = 20, noise: bool = True) -> (torch.Tensor,):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)/tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat


class AugerinoModel(StructBaseLayer):
    def __init__(self, emb_dim=500, max_length=10, h_dim=50, sample_size=1, data_mean=0, data_std=1):
        super(AugerinoModel, self).__init__()
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.h_dim = h_dim
        self.sample_size = sample_size

        self.data_mean = data_mean
        self.data_std = data_std

        # Embedding layer.
        self.emb = nn.Embedding(11, emb_dim, padding_idx=0)

        # Output layer.
        self.out = torch.nn.Linear(h_dim, 1)
        # Register their parameters.
        for num, pp in enumerate(list(self.emb.parameters())):
            self.register_model_parameters('emb_%s' % num, pp)
        for num, pp in enumerate(list(self.out.parameters())):
            self.register_model_parameters('out_%s' % num, pp)

        # One fully connected layer for each max_length.
        self.weights = nn.Parameter(torch.Tensor(
            max_length, self.h_dim, self.emb_dim))
        self.bias = nn.Parameter(torch.Tensor(max_length, self.h_dim))
        self.register_model_parameters('weights', self.weights)
        self.register_model_parameters('bias', self.bias)

        # Structured Linear layer
        self.structure = nn.Parameter(torch.Tensor(max_length, max_length))
        self.register_hyper_parameters('structure', self.structure)

        # Augerino Augmentation param.
        self.permute = nn.Parameter(torch.Tensor(max_length, max_length))
        self.register_model_parameters('permute', self.structure)

        self.reset_parameters()

        # Hard code sharing to identity.
        with torch.no_grad():
            A_init = np.eye(max_length)
            A_init_scale = 10.
            AA = A_init_scale * \
                torch.from_numpy(A_init).float()-(A_init_scale/2.)
            self.structure.copy_(AA)
        self.entropy_weight = 1
        self.tau = 0.1

    def reset_model_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.uniform_(self.bias, -0.1, 0.1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.uniform_(self.bias, -0.1, 0.1)
        nn.init.uniform_(self.structure, -0.1, 0.1)
        nn.init.uniform_(self.permute, -0.1, 0.1)

    def forward_A(self):
        return torch.exp(nn.functional.log_softmax(self.structure, 1))

    def forward(self, x):
        feat = self.emb(x)  # [Batch, max_len, emb_dim]
        # Apply Transformation Here.
        all_copy = []
        for k in range(self.sample_size+1):
            gs_mat = gumbel_sinkhorn(
                torch.cat([self.permute.unsqueeze(0)]*x.shape[0]), tau=self.tau)
            feat = torch.matmul(gs_mat, feat)
            # Structured fully connected layer.
            A = self.forward_A()
            # A on weights is the same as A on activation for mm.
            ww, bb = self.weights, self.bias
            d1, d2, d3 = ww.shape
            # Add sharing.
            ww = A.matmul(ww.reshape(d1, d2*d3))
            ww = ww.reshape(d1, d2, d3)
            bb = A.matmul(bb)
            # Run inference.
            h = []
            for k in range(self.max_length):
                h.append(F.linear(feat[:, k], ww[k], bb[k]))
            hh = torch.stack(h, 1)
            hh = F.relu(hh)
            # Pooling.
            hh = torch.sum(hh, 1)
            # Output layer.
            y_hat = self.out(hh)
            all_copy.append(y_hat.unsqueeze(0))
            # [Batch, 1]
        y_hat = torch.cat(all_copy, 0).mean(0)
        return y_hat

    def predict(self, x):
        y_hat = self.forward(x)
        return y_hat*self.data_std+self.data_mean

    def total_loss(self, input, target):
        return torch.mean(torch.abs(input.squeeze()-(target.squeeze()-self.data_mean)/self.data_std))

    def total_val_loss(self, input, target):
        """Total + reg on A."""
        loss_reg = 0
        AA = self.forward_A()
        loss_reg += torch.trace(torch.sqrt(AA.T.mm(AA)))/AA.shape[0]
        loss_reg += -self.entropy_weight * \
            torch.sum(torch.log(AA+1e-6)*(AA+1e-6), -1).mean()*0.5
        return self.total_loss(input, target) + 0.05*loss_reg
