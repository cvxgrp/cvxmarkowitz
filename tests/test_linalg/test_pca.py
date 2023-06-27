# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from aux.linalg.pca import pca as aux_pca

from cvx.linalg import pca


@pytest.fixture()
def returns(resource_dir):
    prices = pd.read_csv(
        resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    return prices.pct_change().fillna(0.0)


def test_pca(returns):
    xxx = pca(returns)

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
    xxx = pca(returns, n_components=20)
    pd.testing.assert_series_equal(
        xxx.idiosyncratic_returns.std(), pd.Series(np.zeros(20), index=returns.columns)
    )

    assert np.allclose(
        returns.values,
        xxx.factors.values @ xxx.exposure.values + xxx.idiosyncratic_returns.values,
    )


def test_idiosyncratic(returns):
    # as many components as vectors, hence the residual should be zero
    xxx = pca(returns, n_components=15)

    print(xxx.exposure.shape)

    assert np.allclose(
        returns.values,
        xxx.factors.values @ xxx.exposure.values + xxx.idiosyncratic_returns.values,
    )


def test_too_many_factors(returns):
    # as many components as vectors, hence the residual should be zero
    with pytest.raises(ValueError):
        pca(returns, n_components=22)


def test_columns(returns):
    xxx = pca(returns, n_components=15)
    assert xxx.factors.columns.tolist() == list(range(0, 15))
    assert xxx.exposure.columns.tolist() == returns.columns.tolist()
    assert xxx.idiosyncratic_returns.columns.tolist() == returns.columns.tolist()
    assert xxx.cov.columns.tolist() == list(range(0, 15))
    assert xxx.systematic_returns.columns.tolist() == returns.columns.tolist()
    assert xxx.explained_variance.index.tolist() == list(range(0, 15))


def test_alternative(returns):
    xxx = aux_pca(returns, n_components=10)
    xxy = pca(returns, n_components=10)

    pd.testing.assert_index_equal(xxx.asset_names, xxy.asset_names)
    pd.testing.assert_index_equal(xxx.factor_names, xxy.factor_names)

    pd.testing.assert_frame_equal(xxx.cov, xxy.cov)
    pd.testing.assert_frame_equal(xxx.systematic_returns, xxy.systematic_returns)
    pd.testing.assert_frame_equal(xxx.idiosyncratic_returns, xxy.idiosyncratic_returns)
    pd.testing.assert_series_equal(xxx.explained_variance, xxy.explained_variance)
