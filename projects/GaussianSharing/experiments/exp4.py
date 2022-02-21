"""Implements all experiment and baselines."""
import os
import numpy as np
import torch

from gaussian_sharing.data.gaussian_dataset import GaussianDataset
from gaussian_sharing.train import train_loops
from struct_discovery.evaluation.partition_distance import partition_distance


def main():
    # Setup
    num_data = 100
    method_list = ['no_share', 'oracle', 'single_loop']
    # Generated dataset.
    num_trails = 200
    for kdim in [10, 30, 50, 70, 90]:
        for num_train in [30]:
            for a_rank in range(1, 2):
                num_val = num_data - num_train
                results = {m: [] for m in method_list}
                for method in method_list:
                    total_pd = 0
                    for seed in range(0, num_trails):
                        train_val_dataset = GaussianDataset(
                            kdim=kdim, minmax_cluster_num=[a_rank, a_rank],
                            num_samples=num_data, rand_seed=seed)
                        train_dataset = GaussianDataset(
                            kdim=kdim, minmax_cluster_num=[a_rank, a_rank],
                            num_samples=num_train, A=train_val_dataset.A,
                            data=train_val_dataset.data[:num_train])
                        val_dataset = GaussianDataset(
                            kdim=kdim, minmax_cluster_num=[a_rank, a_rank],
                            num_samples=num_val, A=train_val_dataset.A,
                            data=train_val_dataset.data[num_train:])
                        test_dataset = GaussianDataset(
                            kdim=kdim, minmax_cluster_num=[a_rank, a_rank],
                            num_samples=10000, A=train_dataset.A,
                            rand_seed=seed)
                        datasets = (train_dataset, val_dataset, test_dataset)
                        theta_gt = np.expand_dims(test_dataset.theta_gt, 0)
                        A_gt = test_dataset.A_gt

                        A_hat, theta_hat = train_loops.train_one_model(
                            method, datasets)
                        theta_hat = np.expand_dims(theta_hat, 0)
                        # Compute loss.
                        l2_loss = np.sum(
                            np.mean(np.square(test_dataset.data-theta_hat), 0))
                        mse = np.sum(np.square(theta_hat-theta_gt))
                        pd = partition_distance(A_hat, A_gt)
                        results[method].append([l2_loss, mse, pd])
                        print('%s (%s), L2: %s' % (method, seed, l2_loss))
                        print('%s (%s), MSE: %s' % (method, seed, mse))
                        print('%s (%s), PD: %s' % (method, seed, pd))
                        total_pd += pd

                    print(total_pd)
                    np.save('%s/%s_k%s_r%s_t%s_v%s.npy' % (result_out_path, method,
                                                           kdim, a_rank,
                                                           num_train, num_val),
                            results[method])


if __name__ == '__main__':
    # Run experiment 4.
    result_out_path = "_results_exp4/"
    os.makedirs(result_out_path, exist_ok=True)
    main()
