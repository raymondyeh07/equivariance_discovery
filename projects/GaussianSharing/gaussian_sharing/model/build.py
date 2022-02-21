"""Builds from cfg."""
import torch
import torch.nn as nn

from struct_discovery.layers.struct_bias_layer import StructBiasLayer


class StructBiasModel(StructBiasLayer):
    def __init__(self, in_features, train_data=None,
                 A_init=None, A_init_scale=5.,):
        super(StructBiasModel, self).__init__(in_features)
        self.in_features = in_features
        self.A_init = A_init
        self.A_init_scale = A_init_scale
        with torch.no_grad():
            if train_data is not None:
                self.bias.copy_(torch.from_numpy(
                    train_data.mean(0)).float().unsqueeze(-1))
            if self.A_init is not None:
                self.structure.copy_(
                    A_init_scale*torch.from_numpy(A_init).float().T-(A_init_scale/2.))

    def reset_parameters(self):
        nn.init.uniform_(self.bias, -0.1, 0.1)
        nn.init.uniform_(self.structure, -0.01, 0.01)

    def get_A(self):
        return nn.functional.softmax(self.structure, 0).T

    def total_val_loss(self, input, target):
        loss_reg = 0
        AA = self.get_A()
        loss_reg += torch.trace(torch.sqrt(AA.T.mm(AA)))
        loss_reg += -1*torch.sum(torch.log(AA+1e-6)*(AA+1e-6), -1).mean()
        return self.total_loss(input, target) + 1e-2*loss_reg

    @classmethod
    def total_loss(cls, input, target):
        total_loss = torch.nn.functional.mse_loss(input, target)
        return total_loss


def build_model(cfg):
    model = StructBiasModel(cfg.DATASET.NUM_DIMS)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
