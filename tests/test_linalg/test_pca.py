# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.linalg import PCA


@pytest.fixture()
def returns(resource_dir):
    prices = pd.read_csv(
        resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    return prices.pct_change().fillna(0.0).values


def test_pca(returns):
    xxx = PCA(returns=returns, n_components=10)

    assert np.allclose(
        xxx.explained_variance,
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
    # as many components as vectors, hence the residual should be zero
    xxx = PCA(returns=returns, n_components=20)
    np.testing.assert_allclose(xxx.idiosyncratic_returns.std(), np.zeros(20), atol=1e-6)
    # pd.testing.assert_series_equal(
    #    xxx.idiosyncratic_returns.std(), pd.Series(np.zeros(20), index=returns.columns)
    # )

    assert np.allclose(
        returns,
        xxx.factors @ xxx.exposure + xxx.idiosyncratic_returns,
    )


def test_idiosyncratic(returns):
    # as many components as vectors, hence the residual should be zero
    xxx = PCA(returns=returns, n_components=15)

    assert np.allclose(
        returns,
        xxx.factors @ xxx.exposure + xxx.idiosyncratic_returns,
    )


def test_too_many_factors(returns):
    # as many components as vectors, hence the residual should be zero
    with pytest.raises(ValueError):
        PCA(returns=returns, n_components=22)


def test_shape(returns):
    xxx = PCA(returns=returns, n_components=15)
    assert xxx.factors.shape == (320, 15)
    assert xxx.exposure.shape == (15, 20)
    assert xxx.idiosyncratic_returns.shape == (320, 20)
    assert np.std(xxx.idiosyncratic_returns, axis=0).shape == (20,)
    assert xxx.cov.shape == (15, 15)
    assert xxx.systematic_returns.shape == (320, 20)


#
# def test_alternative(returns):
#     xxx = aux_pca(returns, n_components=10)
#     xxy = PCA(returns=returns, n_components=10)
#
#     # pd.testing.assert_index_equal(xxx.asset_names, xxy.asset_names)
#     # pd.testing.assert_index_equal(xxx.factor_names, xxy.factor_names)
#
#     np.testing.assert_allclose(xxx.cov, xxy.cov, atol=1e-10)
#     np.testing.assert_allclose(xxx.systematic_returns, xxy.systematic_returns)
#     np.testing.assert_allclose(xxx.idiosyncratic_returns, xxy.idiosyncratic_returns)
#     np.testing.assert_allclose(xxx.explained_variance, xxy.explained_variance)
#
#     # pd.testing.assert_series_equal(xxx.explained_variance, xxy.explained_variance)
