"""Implements a structed bias layer."""

import torch
import torch.nn as nn

from . import struct_base_layer


class StructBiasLayer(struct_base_layer.StructBaseLayer):
    """Returns a structured bias.
    """

    def __init__(self, in_features):
        super(StructBiasLayer, self).__init__()
        self.in_features = in_features
        self.bias = nn.Parameter(torch.Tensor(in_features, 1))
        self.structure = nn.Parameter(torch.Tensor(in_features, in_features))
        self.register_model_parameters('bias', self.bias)
        self.register_hyper_parameters('structure', self.structure)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.bias, -0.1, 0.1)
        nn.init.uniform_(self.structure, -0.1, 0.1)

    def forward(self, x):
        A = nn.functional.softmax(self.structure, 0)
        mu = self.bias.T.mm(A)
        mu_all = mu.repeat(x.shape[0], 1)
        return mu_all

    def extra_repr(self):
        return 'in_features={}'.format(self.in_features)
