"""Tests for Struct Bias Layer."""
import unittest

import torch
import torch.nn.functional as F

from struct_discovery.layers.struct_bias_layer import StructBiasLayer


class TestStructBiasLayer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.D = 2
        self.model = StructBiasLayer(self.D)
        self.struct_param = list(self.model.hyper_parameters())
        self.model_param = list(self.model.model_parameters())

    def test_check_access(self):
        assert self.struct_param[0].shape[0] == self.D
        assert self.struct_param[0].shape[1] == self.D
        assert self.model_param[0].shape[0] == self.D

    def test_check_forward_shape(self):
        N = 5
        D = 2
        x = torch.zeros(N, D)
        out = self.model(x)
        assert out.shape == x.shape
