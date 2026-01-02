"""Tests for random symmetric positive-definite covariance generator.

Checks basic properties: symmetry and positive definiteness via eigenvalues.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxmarkowitz.linalg.random import rand_cov


@pytest.mark.parametrize("size", [1, 5, 10, 20, 50, 100, 200, 500, 1000])
def test_rand_cov_eigvalsh(size):
    """rand_cov output should be symmetric with strictly positive eigenvalues.

    Uses Hermitian eigensolver for numerical stability.

    Args:
        size: Dimension of the random covariance matrix to generate.
    """
    a = rand_cov(size)
    assert np.allclose(a, np.transpose(a))
    assert np.all(np.linalg.eigvalsh(a) > 0)


@pytest.mark.parametrize("size", [1, 5, 10, 20, 50, 100, 200, 500, 1000])
def test_rand_cov_eigvals(size):
    """Validate positive definiteness using general eigensolver as a cross-check.

    Args:
        size: Dimension of the random covariance matrix to generate.
    """
    a = rand_cov(size)
    assert np.allclose(a, np.transpose(a))
    assert np.all(np.linalg.eigvals(a) > 0)
