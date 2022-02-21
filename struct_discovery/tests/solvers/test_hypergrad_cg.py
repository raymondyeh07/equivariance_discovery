"""Implements test for conjugate gradient."""
import unittest
import torch
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

from struct_discovery.solver.hypergrad.conjugate_gradient import conjugate_gradient


class TestCG(unittest.TestCase):
    def test_check_identity(self):
        A = torch.eye(3)
        def A_func(x): return A.mm(x)
        b = torch.from_numpy(np.array([[1., 2., 3.]], dtype=np.float32)).T
        out = conjugate_gradient(A_func, b).data.numpy()
        np_out = np.linalg.lstsq(A, b, rcond=-1)[0]
        np.testing.assert_array_almost_equal(out.squeeze(), np_out.squeeze())

    def test_check_random(self):
        BB = 0.1*torch.from_numpy(np.random.randn(3, 3)).float()
        A = torch.eye(3) + BB.T.mm(BB)
        def A_func(x): return A.mm(x)
        b = torch.from_numpy(np.array([[1., 2., 3.]], dtype=np.float32)).T
        out = conjugate_gradient(A_func, b, tol=1e-9).data.numpy()
        np_out = np.linalg.lstsq(A, b, rcond=-1)[0]
        sp_out = sp.sparse.linalg.cg(A.numpy(), b.numpy(), atol='legacy')[0]
        np.testing.assert_array_almost_equal(
            out.squeeze(), np_out.squeeze(), decimal=4)
