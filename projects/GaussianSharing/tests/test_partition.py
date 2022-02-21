"""Implements Tests for permutation generation."""
import unittest
from gaussian_sharing.data.helpers import partition


class TestPartition(unittest.TestCase):
    def test_small_case(self):
        out = list(partition([1, 2, 3]))
        for k in [[[1, 2, 3]], [[1], [2, 3]], [[1, 2], [3]], [[2], [1, 3]], [[1], [2], [3]]]:
            assert k in out
