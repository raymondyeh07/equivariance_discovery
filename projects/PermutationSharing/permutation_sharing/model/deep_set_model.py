"""Implements a structured fully connected layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from struct_discovery.layers.struct_base_layer import StructBaseLayer


class DeepSetModel(StructBaseLayer):
    def __init__(self, emb_dim=500, max_length=10, h_dim=100):
        super(DeepSetModel, self).__init__()
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.h_dim = h_dim
        # Embedding layer.
        self.emb = nn.Embedding(11, emb_dim, padding_idx=0)
        # Hidden layer.
        self.layer1 = torch.nn.Linear(emb_dim, h_dim)
        # Output layer.
        self.out = torch.nn.Linear(h_dim, 1)

        # Register all of them to model_params
        for num, pp in enumerate(list(self.emb.parameters())):
            self.register_model_parameters('emb_%s' % num, pp)
        for num, pp in enumerate(list(self.layer1.parameters())):
            self.register_model_parameters('layer1_%s' % num, pp)
        for num, pp in enumerate(list(self.out.parameters())):
            self.register_model_parameters('out_%s' % num, pp)

    def get_A(self):
        """Return all sharing A."""
        AA = torch.zeros(self.max_length, self.max_length)
        AA[:, 0] = 1.
        return AA

    def forward(self, x):
        feat = self.emb(x)  # [Batch, max_len, emb_dim]
        hh = self.layer1(feat)
        hh = torch.tanh(hh)
        # Pooling.
        hh = torch.sum(hh, 1)
        # Output layer.
        y_hat = self.out(hh)
        # [Batch, 1]
        return y_hat

    @classmethod
    def total_loss(cls, input, target):
        return torch.mean(torch.abs(input.squeeze()-target.squeeze()))

    @classmethod
    def total_val_loss(cls, input, target):
        """Same as total_loss."""
        return self.total_loss(input, target)
