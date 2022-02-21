"""Implements a base layer."""

from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn


class StructBaseLayer(nn.Module):
    def __init__(self,):
        super(StructBaseLayer, self).__init__()
        self._model_parameters = OrderedDict()
        self._hyper_parameters = OrderedDict()

    def register_model_parameters(self, name, param):
        self._model_parameters[name] = param

    def register_hyper_parameters(self, name, param):
        self._hyper_parameters[name] = param

    def _model_hyper_parameters(self, is_model=False):
        """Returns an iterator over module parameters."""
        if is_model:
            for name in self._model_parameters:
                yield self._model_parameters[name]
        else:
            for name in self._hyper_parameters:
                yield self._hyper_parameters[name]
        for m in self.children():
            if hasattr(m, '_model_hyper_parameters'):
                for k in m._model_hyper_parameters(is_model):
                    yield k

    def model_parameters(self):
        """Returns an iterator over module parameters."""
        return self._model_hyper_parameters(True)

    def hyper_parameters(self):
        return self._model_hyper_parameters(False)
