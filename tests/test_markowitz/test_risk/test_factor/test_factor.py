# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
from aux.portfolio.min_risk import minrisk_problem
from aux.random import rand_cov

from cvx.linalg import PCA
from cvx.markowitz.risk import FactorModel


@pytest.fixture()
def returns(resource_dir):
    prices = pd.read_csv(
        resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    return prices.pct_change().fillna(0.0).values


def test_timeseries_model(returns):
    # Here we compute the factors and regress the returns on them
    factors = PCA(returns=returns, n_components=10)

    model = FactorModel(assets=20, factors=10)

    model.update(
        cov=factors.cov,
        exposure=factors.exposure,
        idiosyncratic_risk=np.std(factors.idiosyncratic_returns),
    )

    variables = model.variables
    variables["weights"].value = 0.05 * np.ones(20)
    variables["factor_weights"] = model.data["exposure"] @ variables["weights"]

    vola = model.estimate(variables).value
    print(vola)
    np.testing.assert_almost_equal(vola, 0.009233894697646914)


def test_minvar(returns):
    model = FactorModel(assets=20, factors=10)

    problem, _, _ = minrisk_problem(model, variables=model.variables)

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

    variables = model.variables

    np.random.seed(42)

    # define the problem
    # weights, factor_weights = model.variables

    prob, bounds, bounds_factors = minrisk_problem(model, variables)
    assert prob.is_dpp()

    model.update(
        cov=rand_cov(10),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
    )

    bounds.update(
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
    )

    bounds_factors.update(lower_factors=np.zeros(10), upper_factors=np.ones(10))

    prob.solve()
    # assert prob.value == pytest.approx(0.14138117837204583)
    assert np.array(variables["weights"].value[20:]) == pytest.approx(
        np.zeros(5), abs=1e-6
    )

    model.update(
        cov=rand_cov(10),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
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
    assert np.array(variables["weights"].value[20:]) == pytest.approx(
        np.zeros(5), abs=1e-6
    )

    # test that the exposure is correct, e.g. the factor weights match the exposure * asset weights
    assert model.data["exposure"].value @ variables["weights"].value == pytest.approx(
        variables["factor_weights"].value, abs=1e-6
    )

    # test all entries of y are smaller than 0.1
    assert np.all([variables["factor_weights"].value <= 0.1 + 1e-6])
    # test all entries of y are larger than -0.1
    assert np.all([variables["factor_weights"].value >= -(0.1 + 1e-6)])


def test_factor_mini():
    model = FactorModel(assets=3, factors=2)

    variables = model.variables

    assert "weights" in variables
    assert "factor_weights" in variables

    for _, value in variables.items():
        assert type(value) == cp.Variable

    model.update(
        cov=np.eye(2),
        exposure=np.array([[1, 0, 1], [1, 0.5, 1]]),
        idiosyncratic_risk=np.array([0.1, 0.1, 0.1]),
    )

    variables["weights"].value = np.array([0.5, 0.1, 0.2])
    variables["factor_weights"] = model.data["exposure"] @ variables["weights"]

    assert variables["factor_weights"].value == pytest.approx(
        np.array([0.7, 0.75]), abs=1e-6
    )

    residual = np.linalg.norm(np.array([0.05, 0.01, 0.02]))
    systematic = np.linalg.norm(np.array([0.7, 0.75]))

    assert model._residual_risk(variables=variables).value == pytest.approx(residual)
    assert model._systematic_risk(variables=variables).value == pytest.approx(
        systematic
    )

    total = np.linalg.norm(np.array([residual, systematic]))

    assert model.estimate(variables=variables).value == pytest.approx(total)
