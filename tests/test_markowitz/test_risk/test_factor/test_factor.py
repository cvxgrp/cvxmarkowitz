# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.linalg import pca as principal_components
from cvx.markowitz.portfolio.min_risk import minrisk_problem
from cvx.markowitz.risk import FactorModel
from cvx.random import rand_cov


@pytest.fixture()
def returns(resource_dir):
    prices = pd.read_csv(
        resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    return prices.pct_change().fillna(0.0)


def test_timeseries_model(returns):
    # Here we compute the factors and regress the returns on them
    factors = principal_components(returns=returns, n_components=10)

    model = FactorModel(assets=25, factors=10)

    model.update(
        cov=factors.cov.values,
        exposure=factors.exposure.values,
        idiosyncratic_risk=factors.idiosyncratic_returns.std().values,
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )

    w = np.zeros(25)
    w[:20] = 0.05

    vola = model.estimate(w).value
    np.testing.assert_almost_equal(vola, 0.00923407730537884)


def test_minvar(returns):
    model = FactorModel(assets=20, factors=10)

    weights, factor_weights = model.variables

    problem, bounds, bounds_factors = minrisk_problem(
        model, weights, factor_weights=factor_weights
    )

    assert problem.is_dpp()


# def test_data():
#    model = FactorModel(assets=20, factors=10)
#    print(model.data.keys())
#    model.data["lower_factors"].value = 2 * np.ones(10)
#    assert model.data["lower_factors"].value == pytest.approx(
#        model.bounds_factors.data["lower_factors"].value
#    )


def test_estimate_risk():
    """Test the estimate() method"""
    model = FactorModel(assets=25, factors=12)

    np.random.seed(42)

    # define the problem
    weights, factor_weights = model.variables

    prob, bounds, bounds_factors = minrisk_problem(
        model, weights, factor_weights=factor_weights
    )
    assert prob.is_dpp()

    model.update(
        cov=rand_cov(10),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
        # lower_assets=np.zeros(20),
        # upper_assets=np.ones(20),
        # lower_factors=np.zeros(10),
        # upper_factors=np.ones(10),
    )

    bounds.update(
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
    )

    bounds_factors.update(lower_factors=np.zeros(10), upper_factors=np.ones(10))

    prob.solve()
    assert prob.value == pytest.approx(0.14138117837204583)
    assert np.array(weights.value[20:]) == pytest.approx(np.zeros(5), abs=1e-6)

    model.update(
        cov=rand_cov(10),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
        # lower_assets=np.zeros(20),
        # upper_assets=np.ones(20),
        # lower_factors=-0.1 * np.ones(10),
        # upper_factors=0.1 * np.ones(10),
    )

    bounds.update(
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
    )

    bounds_factors.update(
        lower_factors=-0.1 * np.ones(10), upper_factors=0.1 * np.ones(10)
    )

    prob.solve()
    assert prob.value == pytest.approx(0.5454593844618784)
    assert np.array(weights.value[20:]) == pytest.approx(np.zeros(5), abs=1e-6)

    # test that the exposure is correct, e.g. the factor weights match the exposure * asset weights
    assert model.data["exposure"].value @ weights.value == pytest.approx(
        factor_weights.value, abs=1e-6
    )

    # test all entries of y are smaller than 0.1
    assert np.all([factor_weights.value <= 0.1 + 1e-6])
    # test all entries of y are larger than -0.1
    assert np.all([factor_weights.value >= -(0.1 + 1e-6)])
