"""Build optimizers from cfg."""
import torch


def build_lr_scheduler():
    pass


def build_optimizer(cfg, opt_param, lr=0.01, opt_name='ADAM'):
    if opt_name == 'ADAM':
        return torch.optim.Adam(opt_param, lr)
