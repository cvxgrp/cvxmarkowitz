"""Tests for fill utilities that pad vectors/matrices with zeros.

These tests verify that the helper functions keep existing entries and fill
remaining slots with zeros as required.
"""

import numpy as np

from cvx.markowitz.utils.fill import fill_matrix, fill_vector


def test_fill_vector():
    """fill_vector should retain provided values and zero-pad to requested length."""
    a = np.ones(2)
    np.allclose(fill_vector(num=3, x=a), np.array([1, 1, 0]))


def test_fill_matrix():
    """fill_matrix should embed the input block in the top-left and zero-fill the rest."""
    a = np.ones((2, 2))
    np.allclose(fill_matrix(rows=3, cols=3, x=a), np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]))
