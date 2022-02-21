"""Implements partition distance."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def adj_to_set(A):
    num_obj, num_cluster = A.shape
    cluster_all = np.argmax(A, 1)
    all_sets = [set() for _ in range(num_cluster)]
    for obj_idx, cluster_idx in enumerate(cluster_all):
        all_sets[cluster_idx].add(obj_idx)
    return all_sets


def partition_distance(A1, A2):
    S1 = adj_to_set(A1)
    S2 = adj_to_set(A2)
    cost = np.zeros((len(S1), len(S2)))
    for j in range(len(S1)):
        for k in range(len(S2)):
            cost[j, k] = -1*len(S1[j] & S2[k])
    row_ind, col_ind = linear_sum_assignment(cost)
    return len(A1)+cost[row_ind, col_ind].sum()
