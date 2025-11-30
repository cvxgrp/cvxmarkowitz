"""Tests for the MaxSharpe portfolio builder and problem.

These tests cover model wiring, constraints, fixtures, and a small solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.names import ConstraintName as C
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import ModelName as M
from cvx.markowitz.portfolios.max_sharpe import MaxSharpe


@pytest.fixture()
def builder():
    """Construct a MaxSharpe builder fixture with 4 assets."""
    return MaxSharpe(assets=4)


@pytest.fixture()
def problem(builder):
    """Build a cvxpy problem from the MaxSharpe builder with sigma_max set."""
    builder.parameter["sigma_max"].value = 1.0
    return builder.build()


def test_models_builder(builder):
    """Builder should contain expected models (bounds, risk, and return)."""
    assert builder.model.keys() == {M.BOUND_ASSETS, M.RISK, M.RETURN}


def test_constraints(builder):
    """Builder should expose expected constraints including risk cap."""
    assert set(builder.constraints.keys()) == {C.BUDGET, C.LONG_ONLY, C.RISK}


def test_factor_weights(builder):
    """Accessing factor weights without a factor model should raise KeyError."""
    with pytest.raises(KeyError):
        builder.factor_weights


def test_weights(builder):
    """Weights variable should have the correct shape."""
    assert builder.weights.shape == (4,)


def test_is_dpp(problem):
    """Built problem must satisfy DPP rules."""
    assert problem.is_dpp()


def test_models_problem(problem):
    """Problem should carry expected model components (bounds, return, risk)."""
    assert problem.model.keys() == {M.BOUND_ASSETS, M.RETURN, M.RISK}


def test_max_sharpe(solver, problem):
    """Solve a small MaxSharpe instance and validate resulting weights."""
    problem.update(
        **{
            D.CHOLESKY: cholesky(np.array([[1.0, 0.6], [0.6, 2.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.MU: np.array([0.25, 0.30]),
            D.MU_UNCERTAINTY: np.zeros(2),
            D.VOLA_UNCERTAINTY: np.zeros(2),
        }
    )

    problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.weights,
        np.array([5.5556e-01, 4.444e-01, 0.0, 0.0]),
        decimal=4,
    )
