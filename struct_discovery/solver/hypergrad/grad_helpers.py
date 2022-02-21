"""Implements helper functions."""
import torch
from torch.autograd import grad


def my_jacobian(gg, param, v=None):
    out = []
    for grad_k in gg:
        ret = grad(grad_k, param, grad_outputs=v,
                   create_graph=False, retain_graph=True)
        ret = torch.cat([r.view(-1, 1) for r in ret], 0)
        out.append(ret)
    return torch.cat(out, 1)


def flatten_param(param):
    return torch.cat([pp.view(-1, 1) for pp in param], 0)


def compute_grad(batch_data, model, structure=True, create_graph=False,
                 retain_graph=False, allow_unused=False, is_train=True):
    x, y = batch_data
    y_pred = model(x)
    if is_train:
        loss = model.total_loss(y_pred, y)
    else:
        loss = model.total_val_loss(y_pred, y)

    if structure:
        param = model.hyper_parameters()
    else:
        param = model.model_parameters()
    param_grad = grad(loss, param, create_graph=create_graph,
                      retain_graph=retain_graph, allow_unused=allow_unused)
    return flatten_param(param_grad)
