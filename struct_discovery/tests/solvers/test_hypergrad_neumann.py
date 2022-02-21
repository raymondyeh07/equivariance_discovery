"""Implements test for conjugate gradient."""
import unittest
import torch
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

from struct_discovery.solver.hypergrad.neumann_series import neumann_approximation


class TestNeumann(unittest.TestCase):
    def test_check_identity(self):
        A = torch.eye(3)
        def A_func(x): return A.mm(x)
        b = torch.from_numpy(np.array([[1., 2., 3.]], dtype=np.float32)).T
        out = neumann_approximation(A_func, b, maxiter=10).data.numpy()
        np_out = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_array_almost_equal(out.squeeze(), np_out.squeeze())

    def test_check_random(self):
        BB = 0.1*torch.from_numpy(np.random.randn(3, 3)).float()
        A = torch.eye(3) + BB.T.mm(BB)
        def A_func(x): return A.mm(x)
        b = torch.from_numpy(np.array([[1., 2., 3.]], dtype=np.float32)).T
        out = neumann_approximation(
            A_func, b, maxiter=100, alpha=1).data.numpy()
        np_out = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_array_almost_equal(
            out.squeeze(), np_out.squeeze(), 4)
