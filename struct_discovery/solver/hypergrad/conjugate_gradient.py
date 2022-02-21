"""Implements conjugate gradient. Roughly follows python's cg from
https://gist.github.com/sfujiwara/b135e0981d703986b6c2
See http://www.seas.ucla.edu/~vandenbe/236C/lectures/cg.pdf for math.
"""

import torch


def conjugate_gradient(A, b, x0=None, tol=1e-05, maxiter=100, debug=False):
    """
    Use conjugate gradient to solve Ax=b.

    Params
      A: {Function} computes Ax, A(x):=Ax.
      b: {Torch tensor}
      x0: {Torch tensor}
      tol: {Float} Tolerance
      maxiter: {Int} Max iteration.
    """
    if x0 is None:
        x0 = torch.zeros_like(b)
    if maxiter is None:
        maxiter = x0.nelement()*200
    x = x0
    r = A(x) - b.clone()
    p = -r.clone()
    r_k_norm = r.T.mm(r)

    for k in range(maxiter):
        Ap = A(p)
        alpha = r_k_norm / p.T.mm(Ap)
        x += alpha * p
        r += alpha * Ap

        r_kplus1_norm = r.T.mm(r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < tol:
            break
        p = beta * p - r
    return x
