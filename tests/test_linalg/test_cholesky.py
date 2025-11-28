"""Unit tests for Cholesky helper ensuring SPD reconstruction and errors.

Covers basic correctness on symmetric positive-definite matrices and verifies
that non-symmetric or non-square inputs raise linear-algebra errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.linalg.random import rand_cov


def test_cholesky():
    """Verify Cholesky decomposition reconstructs the original SPD matrix."""
    a = rand_cov(10)
    u = cholesky(a)
    # test numpy arrays are equivalent
    assert np.allclose(a, np.transpose(u) @ u)


def test_cholesky_not_symmetric():
    """Non-symmetric matrices should raise LinAlgError."""
    a = np.random.randn(10, 10)
    with pytest.raises(np.linalg.LinAlgError):
        cholesky(a)


def test_cholesky_not_square():
    """Non-square matrices should raise LinAlgError."""
    a = np.random.randn(12, 10)
    with pytest.raises(np.linalg.LinAlgError):
        cholesky(a)


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_cholesky_speed(size):
    """Generate random SPD matrices and verify Cholesky reconstructs covariance.

    Args:
        size: Dimension of the random covariance matrix to test.
    """
    a = rand_cov(size)
    u = cholesky(a)
    assert np.allclose(a, np.transpose(u) @ u)
