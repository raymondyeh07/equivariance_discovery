"""Implements Neuman Series from https://arxiv.org/abs/1911.02590."""

import torch


def neumann_approximation(A, b, maxiter=None, alpha=1, debug=False):
    if maxiter is None:
        maxiter = b.nelement()*200
    v = b.clone()
    p = b.clone()
    for _ in range(maxiter):
        v -= A(v)*alpha
        p += v  # Note the paper has a typo.
    return p
