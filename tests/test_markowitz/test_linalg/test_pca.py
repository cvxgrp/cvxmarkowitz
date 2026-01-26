"""Unit tests for PCA decomposition helper in cvx.linalg.

Covers explained variance values, reconstruction properties, handling of
excess components, and basic performance/shape checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvxmarkowitz.linalg import PCA


@pytest.fixture
def returns(resource_dir):
    """Load prices CSV and return daily returns as a NumPy array.

    Args:
        resource_dir: Pytest fixture providing the path to test resources.
    """
    prices = pd.read_csv(resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True)
    return prices.pct_change().fillna(0.0).values


def test_pca(returns):
    """Explained variance of first 10 components should match known values."""
    pca = PCA(returns=returns, n_components=10)

    assert np.allclose(
        pca.explained_variance,
        np.array(
            [
                0.33383825,
                0.19039608,
                0.11567561,
                0.07965253,
                0.06379108,
                0.04580062,
                0.03461307,
                0.02205145,
                0.01876345,
                0.01757195,
            ]
        ),
    )


def test_idiosyncratic_with_max_factors(returns):
    """With components equal to columns, residual (idiosyncratic) returns vanish."""
    # as many components as vectors, hence the residual should be zero
    pca = PCA(returns=returns, n_components=20)
    assert np.allclose(pca.idiosyncratic_returns.std(), np.zeros(20))


def test_idiosyncratic(returns):
    """Original returns equal systematic + idiosyncratic parts."""
    pca = PCA(returns=returns, n_components=15)

    assert np.allclose(
        returns,
        pca.factors @ pca.exposure + pca.idiosyncratic_returns,
    )


def test_too_many_factors(returns):
    """Requesting more components than columns should raise ValueError."""
    # more components than columns
    with pytest.raises(ValueError, match="number of components cannot exceed"):
        PCA(returns=returns, n_components=22)


@pytest.mark.parametrize("size", [(2, 1), (4, 2), (600, 50), (2000, 100)])
def test_pca_speed(size):
    """Construct PCA for varying matrix sizes and validate shapes.

    Args:
        size: Tuple of (observations, assets) for the random return matrix.
    """
    returns = np.random.randn(size[0], size[1])
    pca = PCA(returns=returns, n_components=size[1])

    assert pca.factors.shape == (size[0], size[1])
    assert pca.exposure.shape == (size[1], size[1])
    assert pca.idiosyncratic_returns.shape == (size[0], size[1])
    assert pca.cov.shape == (size[1], size[1])
    assert pca.systematic_returns.shape == (size[0], size[1])
    assert pca.idiosyncratic_vola.shape == (size[1],)
