"""Tests for factor risk model and factor-aware MinVar builder.

Covers parameter validation, constraint wiring, and DPP compliance, as well
as end-to-end solves with synthetic data.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from cvx.linalg import PCA, cholesky
from cvx.linalg.random import rand_cov
from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import ModelName as M
from cvx.markowitz.portfolios.min_var import MinVar
from cvx.markowitz.risk import FactorModel


@pytest.fixture()
def returns(resource_dir):
    """Load prices CSV and return daily returns as a NumPy array for tests."""
    prices = pd.read_csv(resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True)
    return prices.pct_change().fillna(0.0).values


@pytest.fixture()
def factor_model():
    """Provide a small 3-asset/2-factor FactorModel for validation tests."""
    return FactorModel(assets=3, factors=2)


def test_timeseries_model(returns):
    """Compute PCA factors and validate factor-model risk on synthetic data."""
    # Here we compute the factors and regress the returns on them
    factors = PCA(returns=returns, n_components=10)

    model = FactorModel(assets=20, factors=10)

    model.update(
        **{
            D.CHOLESKY: cholesky(factors.cov),
            D.EXPOSURE: factors.exposure,
            D.IDIOSYNCRATIC_VOLA: factors.idiosyncratic_vola,
            D.SYSTEMATIC_VOLA_UNCERTAINTY: np.zeros(10),
            D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.zeros(20),
        }
    )

    variables = {D.WEIGHTS: cp.Variable(20), D.FACTOR_WEIGHTS: cp.Variable(10)}
    variables[D.WEIGHTS].value = 0.05 * np.ones(20)
    variables[D.FACTOR_WEIGHTS] = model.data["exposure"] @ variables[D.WEIGHTS]
    variables[D._ABS] = cp.abs(variables[D.FACTOR_WEIGHTS])

    vola = model.estimate(variables).value
    np.testing.assert_almost_equal(vola, 0.009233894697646914)


def test_minvar(returns):
    """Build a DPP-compliant MinVar with factor model and verify DPP."""
    problem = MinVar(assets=20, factors=10).build()
    assert problem.is_dpp()


def test_estimate_risk(solver):
    """Test the estimate() method."""
    np.random.seed(42)

    builder = MinVar(assets=25, factors=12)

    problem = builder.build()

    problem.update(
        **{
            D.CHOLESKY: cholesky(rand_cov(10)),
            D.EXPOSURE: np.random.randn(10, 20),
            D.IDIOSYNCRATIC_VOLA: np.random.randn(20),
            D.LOWER_BOUND_ASSETS: np.zeros(20),
            D.UPPER_BOUND_ASSETS: np.ones(20),
            D.LOWER_BOUND_FACTORS: np.zeros(10),
            D.UPPER_BOUND_FACTORS: np.ones(10),
            D.SYSTEMATIC_VOLA_UNCERTAINTY: np.zeros(10),
            D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.zeros(20),
        }
    )

    problem.solve(solver=solver)

    # assert prob.value == pytest.approx(0.14138117837204583)
    assert np.array(problem.weights[20:]) == pytest.approx(np.zeros(5), abs=1e-6)

    problem.update(
        **{
            D.CHOLESKY: cholesky(rand_cov(10)),
            D.EXPOSURE: np.random.randn(10, 20),
            D.IDIOSYNCRATIC_VOLA: np.random.randn(20),
            D.LOWER_BOUND_ASSETS: np.zeros(20),
            D.UPPER_BOUND_ASSETS: np.ones(20),
            D.LOWER_BOUND_FACTORS: -0.1 * np.ones(10),
            D.UPPER_BOUND_FACTORS: +0.1 * np.ones(10),
            D.SYSTEMATIC_VOLA_UNCERTAINTY: np.zeros(10),
            D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.zeros(20),
        }
    )

    problem.solve(solver=solver)

    assert problem.value == pytest.approx(0.5454593844618784, abs=1e-3)
    assert np.array(problem.weights[20:]) == pytest.approx(np.zeros(5), abs=1e-3)

    data = dict(problem.data)

    # test that the exposure is correct, e.g. the factor weights match the exposure * asset weights
    assert data[(M.RISK, "exposure")].value @ problem.weights == pytest.approx(problem.factor_weights, abs=1e-6)

    # test all entries of y are smaller than 0.1
    assert np.all([problem.factor_weights <= 0.1 + 1e-4])
    # test all entries of y are larger than -0.1
    assert np.all([problem.factor_weights >= -(0.1 + 1e-4)])


def test_factor_mini():
    """Verify residual, systematic, and total risk for a tiny factor setup."""
    model = FactorModel(assets=3, factors=2)

    variables = {
        D.WEIGHTS: cp.Variable(3),
        D.FACTOR_WEIGHTS: cp.Variable(2),
        D._ABS: cp.Variable(2),
    }

    model.update(
        **{
            D.CHOLESKY: np.eye(2),
            D.EXPOSURE: np.array([[1, 0, 1], [1, 0.5, 1]]),
            D.IDIOSYNCRATIC_VOLA: np.array([0.1, 0.1, 0.1]),
            D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.array([0.3, 0.3, 0.3]),
            D.SYSTEMATIC_VOLA_UNCERTAINTY: np.array([0.2, 0.1]),
        }
    )

    variables[D.WEIGHTS].value = np.array([0.5, 0.1, 0.2])
    variables[D.FACTOR_WEIGHTS] = model.data["exposure"] @ variables[D.WEIGHTS]
    # Note: dummy is abs(factor_weights)
    variables[D._ABS] = cp.abs(variables[D.FACTOR_WEIGHTS])

    assert variables[D.FACTOR_WEIGHTS].value == pytest.approx(np.array([0.7, 0.75]), abs=1e-6)

    residual = np.sqrt(0.03)
    systematic = np.sqrt(1.098725)

    assert model._residual_risk(variables=variables).value == pytest.approx(residual)
    assert model._systematic_risk(variables=variables).value == pytest.approx(systematic)

    total = np.linalg.norm(np.array([residual, systematic]))

    assert model.estimate(variables=variables).value == pytest.approx(total)


def test_missing_key(factor_model):
    """Updating without a required key should raise CvxError."""
    with pytest.raises(CvxError):
        factor_model.update(
            **{
                D.CHOLESKY: np.eye(2),
                D.IDIOSYNCRATIC_VOLA: np.array([0.1, 0.1, 0.1]),
                D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.array([0.3, 0.3, 0.3]),
                D.EXPOSURE: np.array([[1, 0, 1], [1, 0.5, 1]]),
            }
        )


def test_mismatch_idiosyncratic_vola(factor_model):
    """Idiosyncratic_vola length mismatch vs. assets should raise CvxError."""
    with pytest.raises(CvxError):
        factor_model.update(
            **{
                D.CHOLESKY: np.eye(2),
                D.IDIOSYNCRATIC_VOLA: np.array([0.1, 0.1]),
                D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.array([0.3, 0.3, 0.3]),
                D.EXPOSURE: np.array([[1, 0, 1], [1, 0.5, 1]]),
                D.SYSTEMATIC_VOLA_UNCERTAINTY: np.array([0.2, 0.1]),
            }
        )


def test_mismatch_exposure_idiosyncratic_vola(factor_model):
    """Exposure shape inconsistent with vectors should raise CvxError."""
    with pytest.raises(CvxError):
        factor_model.update(
            **{
                D.CHOLESKY: np.eye(2),
                D.IDIOSYNCRATIC_VOLA: np.array([0.1, 0.1, 0.3]),
                D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.array([0.3, 0.3, 0.3]),
                D.EXPOSURE: np.ones((4, 4)),
                D.SYSTEMATIC_VOLA_UNCERTAINTY: np.array([0.2, 0.1]),
            }
        )


def test_mismatch_systematic_vola(factor_model):
    """Systematic_vola_uncertainty length mismatch vs. factors should error."""
    with pytest.raises(CvxError):
        factor_model.update(
            **{
                D.CHOLESKY: np.eye(2),
                D.IDIOSYNCRATIC_VOLA: np.array([0.1, 0.1, 0.1]),
                D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.array([0.3, 0.3, 0.3]),
                D.EXPOSURE: np.array([[1, 0, 1], [1, 0.5, 1]]),
                D.SYSTEMATIC_VOLA_UNCERTAINTY: np.array([0.2]),
            }
        )


def test_mismatch_matrix(factor_model):
    """Cholesky shape inconsistent with exposure matrix should raise CvxError."""
    with pytest.raises(CvxError):
        factor_model.update(
            **{
                D.CHOLESKY: np.eye(1),
                D.IDIOSYNCRATIC_VOLA: np.array([0.1, 0.1, 0.1]),
                D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.array([0.3, 0.3, 0.3]),
                D.EXPOSURE: np.array([[1, 0, 1], [1, 0.5, 1]]),
                D.SYSTEMATIC_VOLA_UNCERTAINTY: np.array([0.2, 0.3]),
            }
        )
