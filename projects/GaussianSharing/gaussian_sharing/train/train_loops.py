"""Implemenets helpers for analtyical solution."""

import numpy as np
import torch
from gaussian_sharing.model.analytical_model import AnalyticStructBiasModel
from gaussian_sharing.model.build import StructBiasModel

from struct_discovery.solver.hypergrad import implicit_function

from gaussian_sharing.data import helpers


def train_one_model(method, datasets):
    assert method in ['no_share', 'oracle',
                      'brute_force', 'single_loop', 'double_loop']
    train_dataset, val_dataset, test_dataset = datasets
    if method == 'no_share':
        A_hat, theta_hat = fit_no_share(train_dataset, val_dataset)
    if method == 'oracle':
        A_hat, theta_hat = fit_oracle(train_dataset, val_dataset)
    if method == 'brute_force':
        A_hat, theta_hat = fit_brute_force(train_dataset, val_dataset)
    if method == 'single_loop':
        best_in_sample_loss = np.inf
        # Non-convex optimization, train with different random initialization.
        num_init = 3
        for k in range(num_init):
            A_hat_k, theta_hat_k, in_sample_loss = fit_single_loop(
                train_dataset, val_dataset)
            if in_sample_loss < best_in_sample_loss:
                best_in_sample_loss = in_sample_loss
                A_hat = A_hat_k
                theta_hat = theta_hat_k

    if method == 'double_loop':
        A_hat, theta_hat = fit_double_loop(train_dataset, val_dataset)
    return A_hat, theta_hat


def fit_no_share(train_dataset, val_dataset):
    train_data = train_dataset.data
    val_data = val_dataset.data
    kdim = train_data.shape[-1]
    A_ind = np.eye(kdim)
    train_val_data = np.concatenate([train_data, val_data], 0)
    kdim = train_data.shape[-1]
    model = AnalyticStructBiasModel(
        kdim, train_val_data, A_init=A_ind, A_init_scale=10)
    theta_hat = model.forward(torch.zeros([1, kdim]))
    return A_ind, theta_hat.data.cpu().squeeze().numpy()


def fit_oracle(train_dataset, val_dataset):
    A_gt = train_dataset.A_gt
    train_data = train_dataset.data
    val_data = val_dataset.data
    train_val_data = np.concatenate([train_data, val_data], 0)
    kdim = train_data.shape[-1]
    model = AnalyticStructBiasModel(
        kdim, train_val_data, A_init=A_gt, A_init_scale=10)
    theta_hat = model.forward(torch.zeros([1, kdim]))
    return A_gt, theta_hat.data.cpu().squeeze().numpy()


def fit_single_loop(train_dataset, val_dataset, config=None):
    defaults = {'num_iter': 1000, 'outer_lr': 2e-2}
    if config is not None:
        pass  # TODO: update with config.

    num_iter = defaults['num_iter']
    lr = defaults['outer_lr']

    kdim = train_dataset.data.shape[-1]
    model = AnalyticStructBiasModel(kdim, train_dataset.data)
    hyper_optimizer = torch.optim.RMSprop(
        model.hyper_parameters(), lr, weight_decay=1e-4)

    torch_val_data = torch.from_numpy(val_dataset.data).float()

    model.train()
    model.cuda()
    torch_val_data = torch_val_data.cuda()

    for _ in range(num_iter):
        y_pred = model.forward(torch_val_data)
        AA = model.get_A()
        loss_reg = 0
        loss_reg += torch.trace(torch.sqrt(AA.T.mm(AA)))
        loss_reg += -1*torch.sum(torch.log(AA+1e-6)*(AA+1e-6), -1).mean()
        loss = model.total_val_loss(
            y_pred, torch_val_data) + 1e-2*loss_reg
        model.zero_grad()
        loss.backward()
        hyper_optimizer.step()
    A_best_idx = model.get_A().data.cpu().numpy().argmax(-1)
    A_best = np.zeros((kdim, kdim))
    A_best[np.arange(0, kdim), A_best_idx] = 1
    # Refit for the best-parameters.
    train_data = train_dataset.data
    val_data = val_dataset.data
    train_val_data = np.concatenate([train_data, val_data], 0)
    model = AnalyticStructBiasModel(
        kdim, train_val_data, A_init=A_best, A_init_scale=10)
    theta_hat = model.forward(torch.zeros([1, kdim]))[0]
    return A_best, theta_hat.data.cpu().squeeze().numpy(), loss.item()


def fit_double_loop(train_dataset, val_dataset, config=None):
    defaults = {'num_iter': 1000, 'num_inner': 10,
                'outer_lr': 1e-2, 'inner_lr': 1e-4}
    if config is not None:
        pass  # TODO: update with config.

    num_iter = defaults['num_iter']
    num_inner = defaults['num_inner']
    lr = defaults['outer_lr']
    inner_lr = defaults['inner_lr']

    kdim = train_dataset.data.shape[-1]
    model = StructBiasModel(kdim, train_dataset.data)
    model_optimizer = torch.optim.Adam(model.model_parameters(), inner_lr)
    hyper_optimizer = torch.optim.Adam(model.hyper_parameters(), lr)

    torch_train_data = torch.from_numpy(train_dataset.data).float()
    torch_val_data = torch.from_numpy(val_dataset.data).float()

    model.train()
    model.cuda()
    torch_val_data = torch_val_data.cuda()
    torch_train_data = torch_train_data.cuda()

    for _ in range(num_iter):
        # Inner loop.
        for inner in range(num_inner):
            y_pred = model.forward(torch_train_data)
            inner_loss = model.total_loss(y_pred, torch_train_data)
            model.zero_grad()
            inner_loss.backward()
            model_optimizer.step()
        # Outer loop.
        y_pred = model.forward(torch_val_data)
        # Computes hypergradient.
        model.zero_grad()
        hyper_grad = implicit_function.compute_hypergrad(
            (torch_train_data, torch_train_data),
            (torch_val_data, torch_val_data),
            model,
            method='EXACT')
        # Update hypergradient.
        with torch.no_grad():
            bidx = 0
            for mm in model.hyper_parameters():
                mm_size = mm.nelement()
                eidx = bidx + mm_size
                mm.grad = torch.reshape(
                    hyper_grad[bidx:eidx, :], mm.shape).clone()
        hyper_optimizer.step()
    A_best_idx = model.get_A().data.cpu().numpy().argmax(-1)
    A_best = np.zeros((kdim, kdim))
    A_best[np.arange(0, kdim), A_best_idx] = 1

    # Refit for the best-parameters.
    train_data = train_dataset.data
    val_data = val_dataset.data
    train_val_data = np.concatenate([train_data, val_data], 0)
    model = AnalyticStructBiasModel(kdim, train_val_data, A_init=A_best)
    theta_hat = model.forward(torch.zeros([1, kdim]))[0]
    return A_best, theta_hat.data.cpu().squeeze().numpy()


def fit_brute_force(train_dataset, val_dataset):
    # Search over all A.
    train_data = train_dataset.data
    val_data = val_dataset.data
    train_val_data = np.concatenate([train_data, val_data], 0)
    torch_val_data = torch.from_numpy(val_dataset.data).float()

    kdim = train_data.shape[-1]
    group = list(helpers.partition(list(range(0, kdim))))
    min_val_loss = np.inf
    best_A = None
    best_theta = None
    for curr_group in group:
        A = np.zeros((kdim, kdim))
        for cnum, cluster in enumerate(curr_group):
            A[cnum, cluster] = 1
        A = A.T
        # Fit on train given A.
        model = AnalyticStructBiasModel(kdim, train_data, A_init=A)
        y_pred = model.forward(torch_val_data)
        val_loss = model.total_val_loss(y_pred, torch_val_data)
        AA = model.get_A()
        loss_reg = 0
        loss_reg += 1e-2*torch.trace(torch.sqrt(AA.T.mm(AA)))
        val_loss += loss_reg
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            best_A = A
            best_theta = y_pred[0:1]
    # Refit using the best A.
    model = AnalyticStructBiasModel(
        kdim, train_val_data, A_init=best_A, A_init_scale=10.)
    best_theta = model.forward(torch_val_data)[0:1]
    return best_A, best_theta.data.cpu().squeeze().numpy()
