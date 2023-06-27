# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
from aux.portfolio.min_var import MinVar
from aux.random import rand_cov

from cvx.linalg import cholesky
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
        chol=cholesky(factors.cov),
        exposure=factors.exposure,
        idiosyncratic_risk=factors.idiosyncratic_risk,
    )

    variables = model.variables
    variables["weights"].value = 0.05 * np.ones(20)
    variables["factor_weights"] = model.data["exposure"] @ variables["weights"]

    vola = model.estimate(variables).value
    np.testing.assert_almost_equal(vola, 0.009233894697646914)


def test_minvar(returns):
    problem = MinVar(assets=20, factors=10).build()
    assert problem.is_dpp()


def test_estimate_risk():
    """Test the estimate() method"""
    np.random.seed(42)

    builder = MinVar(assets=25, factors=12)

    builder.update(
        chol=cholesky(rand_cov(10)),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )

    problem = builder.build()
    problem.solve()

    # assert prob.value == pytest.approx(0.14138117837204583)
    assert np.array(builder.variables["weights"].value[20:]) == pytest.approx(
        np.zeros(5), abs=1e-6
    )

    builder.update(
        chol=cholesky(rand_cov(10)),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=-0.1 * np.ones(10),
        upper_factors=0.1 * np.ones(10),
    )

    problem.solve()

    assert problem.value == pytest.approx(0.5454593844618784)
    assert np.array(builder.variables["weights"].value[20:]) == pytest.approx(
        np.zeros(5), abs=1e-6
    )

    data = dict(builder.data)

    # test that the exposure is correct, e.g. the factor weights match the exposure * asset weights
    assert data[("risk", "exposure")].value @ builder.variables[
        "weights"
    ].value == pytest.approx(builder.variables["factor_weights"].value, abs=1e-6)

    # test all entries of y are smaller than 0.1
    assert np.all([builder.variables["factor_weights"].value <= 0.1 + 1e-6])
    # test all entries of y are larger than -0.1
    assert np.all([builder.variables["factor_weights"].value >= -(0.1 + 1e-6)])


def test_factor_mini():
    model = FactorModel(assets=3, factors=2)

    variables = model.variables

    assert "weights" in variables
    assert "factor_weights" in variables

    for _, value in variables.items():
        assert type(value) == cp.Variable

    model.update(
        chol=np.eye(2),
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
