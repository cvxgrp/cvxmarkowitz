"""Unit tests for PCA decomposition helper in cvx.linalg.

Covers explained variance values, reconstruction properties, handling of
excess components, and basic performance/shape checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.linalg import pca


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
    result = pca(returns, n_components=10)

    assert np.allclose(
        result.explained_variance,
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
    result = pca(returns, n_components=20)
    assert np.allclose(np.std(result.idiosyncratic, axis=0), np.zeros(20))


def test_idiosyncratic(returns):
    """Original returns equal systematic + idiosyncratic parts."""
    result = pca(returns, n_components=15)

    assert np.allclose(returns, result.systematic + result.idiosyncratic)


def test_too_many_factors(returns):
    """When n_components exceeds the number of assets, pca clamps to n_assets.

    The new cvx.linalg.pca API silently caps n_components at min(n_samples,
    n_assets) rather than raising an error.  This test documents that
    intentional behaviour: the output shape reflects the actual rank, not the
    requested component count.
    """
    n_assets = returns.shape[1]  # 20
    result = pca(returns, n_components=n_assets + 5)

    # Output is clamped to the available rank – no exception raised.
    assert result.factors.shape[1] <= n_assets
    assert result.exposure.shape[0] <= n_assets


@pytest.mark.parametrize("size", [(4, 2), (600, 50), (2000, 100)])
def test_pca_speed(size):
    """Construct PCA for varying matrix sizes and validate shapes.

    Args:
        size: Tuple of (observations, assets) for the random return matrix.
    """
    returns = np.random.randn(size[0], size[1])
    result = pca(returns, n_components=size[1])

    assert result.factors.shape == (size[0], size[1])
    assert result.exposure.shape == (size[1], size[1])
    assert result.idiosyncratic.shape == (size[0], size[1])
    assert result.cov.shape == (size[1], size[1])
    assert result.systematic.shape == (size[0], size[1])
    assert np.std(result.idiosyncratic, axis=0).shape == (size[1],)
