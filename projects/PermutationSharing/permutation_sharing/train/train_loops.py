"""Implements helpers for training."""

import copy
import numpy as np
import torch

from torch.utils.data import DataLoader, ConcatDataset
from permutation_sharing.model.struct_fc_model import StructFCModel
from permutation_sharing.model.deep_set_model import DeepSetModel
from struct_discovery.solver.hypergrad import implicit_function

from struct_discovery.evaluation.partition_distance import partition_distance


def get_dataloader(input_dataset, batch_size=128, shuffle=True, drop_last=True):
    input_dataloader = DataLoader(input_dataset, batch_size=batch_size,
                                  shuffle=shuffle, drop_last=drop_last)
    return input_dataloader


def create_loader(method, datasets):
    train_dataset, val_dataset, hyper_dataset = datasets
    if method in ['oracle', 'deep_set', 'no_share']:
        # Oracle doesn't search for A, directly use all data
        train_val_dataset = ConcatDataset([train_dataset, val_dataset])
        train_loader = get_dataloader(train_val_dataset, batch_size=len(
            train_val_dataset), shuffle=True, drop_last=True)
        val_loader = None
    else:
        train_loader = get_dataloader(train_dataset, batch_size=len(
            train_dataset), shuffle=True, drop_last=True)
        val_loader = get_dataloader(val_dataset, batch_size=len(
            val_dataset), shuffle=True, drop_last=True)
    # hyper_dataset is for all other hyperparameters (learning stopping, learning rate, etc.)
    hyper_loader = get_dataloader(hyper_dataset, batch_size=len(
        hyper_dataset), shuffle=False, drop_last=False)
    return train_loader, val_loader, hyper_loader


def get_data_mean_std(datasets):
    train_dataset, val_dataset, _ = datasets
    train_label = train_dataset.label
    val_label = val_dataset.label
    data_label = np.concatenate([train_label, val_label], 0)
    return data_label.mean(), data_label.std()


def fit_one_model(method, datasets):
    train_loader, val_loader, hyper_loader = create_loader(method, datasets)
    # Compute mean and std from data.
    data_mean, data_std = get_data_mean_std(datasets)
    hyper_params = {'lr': 1e-3, 'num_epoch': 100000,
                    'patience': 1000, 'data_mean': data_mean, 'data_std': data_std}
    if method == 'oracle':
        model = fit_oracle(train_loader, val_loader,
                           hyper_loader, hyper_params)

    if method == 'deep_set':
        model = fit_deep_set(train_loader, val_loader,
                             hyper_loader, hyper_params)

    elif method == 'no_share':
        model = fit_no_share(train_loader, val_loader,
                             hyper_loader, hyper_params)

    elif method == 'ours':
        hyper_params['num_epoch'] = 500
        hyper_params['lr'] = 1e-3
        hyper_params['outer_lr'] = 1e-2
        hyper_params['num_inner'] = 250
        hyper_params['patience'] = 100
        val_loss_best = np.inf
        if train_loader.dataset.A_rank == 1:
            # Non-convex optimization try random initialization.
            init_modes = ['uniform', 'uniform1', 'uniform2']
        else:
            init_modes = ['uniform', 'uniform1', 'uniform2']

        for init_mode in init_modes:
            A_best, _ = fit_ours_double_loop(
                train_loader, val_loader, hyper_loader, hyper_params, init_mode)
            torch.manual_seed(1)
            model = StructFCModel(max_length=hyper_loader.dataset.max_length,
                                  A_init=A_best,
                                  data_mean=hyper_params['data_mean'],
                                  data_std=hyper_params['data_std'])
            hyper_fit_params = {'lr': 1e-3,
                                'num_epoch': 100000, 'patience': 1000}
            model = fit_standard_model(
                train_loader, val_loader, model, hyper_fit_params)
            val_loss = eval_model(model, val_loader)
            # Check on val to prevent underfitting of A.
            if val_loss_best > val_loss:
                val_loss_best = val_loss
                A_best_best = A_best

        # Finally, retrain on all the data.
        torch.manual_seed(2)
        model = StructFCModel(max_length=hyper_loader.dataset.max_length,
                              A_init=A_best_best,
                              data_mean=hyper_params['data_mean'],
                              data_std=hyper_params['data_std'])
        train_val_dataset = ConcatDataset(
            [train_loader.dataset, val_loader.dataset])
        train_val_loader = get_dataloader(train_val_dataset, batch_size=len(
            train_val_dataset), shuffle=True, drop_last=True)
        hyper_fit_params = {'lr': 1e-3, 'num_epoch': 100000, 'patience': 1000}
        model = fit_standard_model(
            train_val_loader, hyper_loader, model, hyper_fit_params)
    return model


def fit_oracle(train_loader, val_loader, hyper_loader, hyper_params):
    assert val_loader is None
    torch.manual_seed(0)
    model = StructFCModel(max_length=hyper_loader.dataset.max_length, A_init=hyper_loader.dataset.A,
                          data_mean=hyper_params['data_mean'],
                          data_std=hyper_params['data_std'])
    best_model = fit_standard_model(
        train_loader, hyper_loader, model, hyper_params)
    return best_model


def fit_deep_set(train_loader, val_loader, hyper_loader, hyper_params):
    assert val_loader is None
    A = np.zeros_like(hyper_loader.dataset.A)
    A[0, :] = 1.
    torch.manual_seed(0)
    model = StructFCModel(max_length=hyper_loader.dataset.max_length, A_init=A,
                          data_mean=hyper_params['data_mean'],
                          data_std=hyper_params['data_std'])
    best_model = fit_standard_model(
        train_loader, hyper_loader, model, hyper_params)
    return best_model


def fit_no_share(train_loader, val_loader, hyper_loader, hyper_params):
    assert val_loader is None
    torch.manual_seed(0)
    model = StructFCModel(max_length=hyper_loader.dataset.max_length,
                          A_init=np.eye(hyper_loader.dataset.max_length),
                          data_mean=hyper_params['data_mean'],
                          data_std=hyper_params['data_std'])
    best_model = fit_standard_model(
        train_loader, hyper_loader, model, hyper_params)
    return best_model


def fit_ours_double_loop(train_loader, val_loader, hyper_loader, hyper_params, init_mode):
    inner_lr = hyper_params['lr']
    outer_lr = hyper_params['outer_lr']
    num_epoch = hyper_params['num_epoch']
    num_inner = hyper_params['num_inner']
    patience = hyper_params['patience']
    warmup = 30
    max_len = hyper_loader.dataset.max_length
    torch.manual_seed(0)
    model = StructFCModel(max_length=max_len,
                          data_mean=hyper_params['data_mean'],
                          data_std=hyper_params['data_std'])

    model.cuda()
    if init_mode == 'uniform2':
        with torch.no_grad():
            torch.nn.init.uniform_(model.structure, -0.1, 0.1)
    if init_mode == 'uniform1':
        with torch.no_grad():
            model.structure.data *= -1

    model_optimizer = torch.optim.AdamW(
        model.model_parameters(), inner_lr, weight_decay=1e-3)
    hyper_optimizer = torch.optim.Adam(model.hyper_parameters(), outer_lr)
    hyper_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        hyper_optimizer, 'min', factor=0.5, patience=50, cooldown=0)

    epoch_count = 0
    x_hyper, y_hyper = hyper_loader.dataset.data, hyper_loader.dataset.label
    x_val, y_val = val_loader.dataset.data, val_loader.dataset.label
    x, y = train_loader.dataset.data, train_loader.dataset.label

    x_hyper, y_hyper = torch.from_numpy(
        x_hyper).cuda(), torch.from_numpy(y_hyper).cuda()
    x_val, y_val = torch.from_numpy(
        x_val).cuda(), torch.from_numpy(y_val).cuda()
    x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

    best_val_loss = np.inf
    for k in range(num_epoch):
        # Inner loop.
        for _ in range(num_inner):
            y_pred = model.forward(x)
            inner_loss = model.total_loss(y_pred, y)
            model.zero_grad()
            inner_loss.backward()
            model_optimizer.step()

        # Outer loop.
        # Computes hypergradient.
        hyper_grad = implicit_function.compute_hypergrad(
            (x, y),
            (x_val, y_val),
            model,
            hyper_optimizer,
            method='NEUMANN')
        model.zero_grad()
        # Update hypergradient.
        with torch.no_grad():
            bidx = 0
            for mm in model.hyper_parameters():
                mm_size = mm.nelement()
                eidx = bidx + mm_size
                mm.grad = torch.reshape(
                    hyper_grad[bidx:eidx, :], mm.shape).clone()
        torch.nn.utils.clip_grad_norm_(model.hyper_parameters(), 25)
        hyper_optimizer.step()
        if patience == 0:
            break
        patience -= 1
        # Early stopping check on hyper set.
        with torch.no_grad():
            y_pred = model.forward(x_val)
            outer_loss = model.total_val_loss(y_pred, y_val).item()

        if k >= warmup:
            hyper_scheduler.step(outer_loss)

        if best_val_loss > outer_loss*1.0001 and k >= warmup:  # Warmup
            print('Epoch: %s' % k)
            print(outer_loss)
            print(model.forward_A().argmax(1))
            best_val_loss = outer_loss
            best_model = copy.deepcopy(model)
            # reset patience
            patience = hyper_params['patience']

    # Get A_val
    A_best_idx = best_model.forward_A().data.cpu().numpy().argmax(-1)
    A_best = np.zeros((max_len, max_len))
    A_best[np.arange(0, max_len), A_best_idx] = 1
    return A_best, best_val_loss


def fit_standard_model(train_loader, hyper_loader, model, hyper_params):
    torch.manual_seed(0)
    model.cuda()
    # Optimizer.
    lr = hyper_params['lr']
    num_epoch = hyper_params['num_epoch']
    patience = hyper_params['patience']
    model_optimizer = torch.optim.Adam(model.model_parameters(), lr,)
    model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model_optimizer, 'min', factor=0.25, patience=1000)

    best_hyper_loss = np.inf
    best_model = None
    for epoch in range(num_epoch):
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            y_hat = model(x)
            model.zero_grad()
            train_loss = model.total_loss(y_hat, y)
            train_loss.backward()
            model_optimizer.step()
        model_scheduler.step(train_loss)
        # Evaluation + logging here.
        hyper_loss = eval_model(model, hyper_loader)
        if patience == 0:
            break
        patience -= 1
        if best_hyper_loss > hyper_loss*1.0001:
            # Save the best model.
            best_hyper_loss = hyper_loss
            best_model = copy.deepcopy(model)
            patience = hyper_params['patience']
    return best_model


def eval_model(model, data_loader, is_test=False):
    total_loss = 0
    total_mae_loss = 0
    total_mse_loss = 0
    total_acc = 0
    total_pd = 0
    count = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda()
            y = y.cuda()
            if is_test:
                y_hat = model.predict(x)
            else:
                y_hat = model(x)
            loss = model.total_loss(y_hat, y).item()
            total_loss += loss*y_hat.shape[0]
            count += y.shape[0]
            if is_test:
                total_mae_loss += torch.sum(torch.abs(y_hat.squeeze() -
                                                      y.squeeze())).item()
                total_mse_loss += torch.sum((y_hat.squeeze() -
                                             y.squeeze())**2).item()
                total_acc += torch.sum(torch.round(y_hat)
                                       == y.int()).float().item()
    if is_test:
        total_mae_loss = total_mae_loss/count
        total_mse_loss = total_mse_loss/count
        total_acc = total_acc/count
        A_best_idx = model.forward_A().data.cpu().numpy().argmax(-1)
        A_best = np.zeros_like(data_loader.dataset.A)
        A_best[np.arange(0, A_best.shape[0]), A_best_idx] = 1
        total_pd = partition_distance(A_best, data_loader.dataset.A)
        return total_mae_loss, total_mse_loss, total_acc, total_pd
    else:
        return total_loss/count
