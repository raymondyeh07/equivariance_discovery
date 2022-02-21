"""Test for simple trainer"""
import unittest

import numpy as np
import torch
import torch.nn as nn

from struct_discovery.engine.trainers.simple_trainer import SimpleTrainer


class TestSimpleTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        x = torch.zeros(10, 2)
        y = torch.zeros(10, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False)

        class LinearModel(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, 1)

            def forward(self, x):
                xx, yy = x
                ret = {}
                pred = self.linear(xx)
                loss = torch.sum((pred-yy)**2)
                ret['loss'] = loss
                return ret
        self.model = LinearModel(2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.1)

    def test_build_simple_trainer(self):
        trainer = SimpleTrainer(self.model, self.data_loader, self.optimizer)
        trainer.train(0, 5)
        assert len(trainer.storage._history['loss']._data) == 5
