"""Implements a closed form model."""
import torch
import torch.nn as nn

from struct_discovery.layers.struct_bias_layer import StructBiasLayer


class AnalyticStructBiasModel(StructBiasLayer):
    def __init__(self, in_features, train_data, A_init=None,
                 A_init_scale=5., method='closed'):
        super(AnalyticStructBiasModel, self).__init__(in_features)
        self.in_features = in_features
        self.A_init = A_init
        with torch.no_grad():
            self.bias.copy_(torch.from_numpy(
                train_data.mean(0)).float().unsqueeze(-1))
            if self.A_init is not None:
                self.structure.copy_(
                    A_init_scale*torch.from_numpy(A_init).float().T-(A_init_scale/2.))
        self.method = method

    def reset_parameters(self):
        nn.init.uniform_(self.bias, -0.1, 0.1)
        nn.init.normal_(self.structure, 0, 0.1)

    def forward(self, x):
        A_t = torch.exp(nn.functional.log_softmax(self.structure, 0))
        A_norm = torch.sum(A_t.T, 0)
        A_bar = A_t.T/(A_norm+1e-8)
        # Solve for Psi
        theta = self.bias.T
        if self.method == 'closed':
            # A_bar of the form derived in Claim 1.
            mu = theta.mm(A_bar.mm(A_t))
        else:
            raise ValueError('Unsupported method: %s' % self.method)
        mu_all = mu.repeat(x.shape[0], 1)
        return mu_all

    def get_A(self):
        return torch.exp(nn.functional.log_softmax(self.structure, 0)).T

    @classmethod
    def total_loss(cls, input, target):
        total_loss = torch.nn.functional.mse_loss(input, target)
        return total_loss

    @classmethod
    def total_val_loss(cls, input, target):
        return cls.total_loss(input, target)
