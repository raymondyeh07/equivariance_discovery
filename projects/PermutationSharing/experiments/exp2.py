"""Experiment for variant sum of numbers dataset."""
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from permutation_sharing.data.numbers_sum_dataset import NumbersSumDataset
from permutation_sharing.train.train_loops import fit_one_model, eval_model, get_dataloader

from tqdm import tqdm


def main():
    num_train = 100  # Train and val is the standard training set.
    num_val = 150
    # This is the standard validation set. To search hyperparams.
    num_hyper = 250
    num_test = 100000

    num_runs = 5
    a_rank = 2
    if a_rank == 1:
        w_scale = 1
    if a_rank == 2:
        w_scale = -1  # For rank 2 experiment.

    add_noise = 0.5
    method_list = ['no_share', 'oracle', 'ours', 'augerino3']
    for max_length in [2, 4, 6, 8, 10]:
        results = {m: [] for m in method_list}
        for method in method_list:
            for rand_seed in tqdm(range(0, num_runs)):
                # Create Datasets
                train_set = NumbersSumDataset(
                    num_train, max_length=max_length, A_rank=a_rank,
                    add_noise=add_noise, rand_seed=rand_seed, w_scale=w_scale)
                A_gt = train_set.A
                val_set = NumbersSumDataset(
                    num_val, max_length=max_length, A_rank=a_rank,
                    A=A_gt, add_noise=add_noise,
                    rand_seed=rand_seed+num_runs, w_scale=w_scale)
                hyper_set = NumbersSumDataset(
                    num_hyper, max_length=max_length, A_rank=a_rank,
                    A=A_gt, add_noise=add_noise,
                    rand_seed=rand_seed+2*num_runs, w_scale=w_scale)
                test_set = NumbersSumDataset(num_test, max_length=max_length,
                                             A_rank=a_rank,
                                             A=A_gt, add_noise=False,
                                             rand_seed=rand_seed+3*num_runs,
                                             w_scale=w_scale)
                # Train the models.
                datasets = (train_set, val_set, hyper_set)
                model = fit_one_model(method, datasets)
                # Evaluate the models.
                test_loader = get_dataloader(
                    test_set, shuffle=False, drop_last=False)
                metrices = eval_model(model, test_loader, is_test=True)
                print(metrices)
                results[method].append(metrices)
            # Save to file.
            result_out_path = "_results_exp2"
            os.makedirs(result_out_path, exist_ok=True)
            np.save('%s/%s_k_%s_r_%s_t_%s_v_%s_n_%s' % (result_out_path,
                                                        method, max_length,
                                                        a_rank, num_train,
                                                        num_val, add_noise),
                    results[method])


if __name__ == '__main__':
    main()
