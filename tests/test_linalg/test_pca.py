# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
