"""Tests for the SoftRisk portfolio builder and problem.

These tests cover model wiring, constraints, variables, parameters, the
soft-risk objective/CallbackParam, and a small solve.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
from cvx.linalg import cholesky

from cvxmarkowitz.names import ConstraintName as C
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.names import ParameterName as P
from cvxmarkowitz.portfolios.soft_risk import SoftRisk


@pytest.fixture
def builder():
    """Construct a SoftRisk builder fixture with 4 assets."""
    return SoftRisk(assets=4)


@pytest.fixture
def problem(builder):
    """Build a cvxpy problem from the SoftRisk builder."""
    return builder.build()


def test_models_builder(builder):
    """Builder should contain expected models (bounds, risk, and return)."""
    assert builder.model.keys() == {M.BOUND_ASSETS, M.RISK, M.RETURN}


def test_constraints(builder):
    """Builder should expose long-only, budget, risk, and max_risk constraints."""
    assert set(builder.constraints.keys()) == {C.LONG_ONLY, C.BUDGET, C.RISK, "max_risk"}


def test_parameters(builder):
    """Builder should declare the soft-risk parameters."""
    for name in (P.SIGMA_MAX, P.SIGMA_TARGET, P.OMEGA):
        assert name in builder.parameter


def test_objective_is_maximize(builder):
    """The soft-risk objective should be a maximization."""
    assert isinstance(builder.objective, cp.Maximize)


def test_weights(builder):
    """Weights variable should have the correct shape."""
    assert builder.weights.shape == (4,)


def test_is_dpp(problem):
    """Built problem must satisfy DPP rules."""
    assert problem.is_dpp()


def test_models_problem(problem):
    """Problem should carry expected model components (bounds, return, risk)."""
    assert problem.model.keys() == {M.BOUND_ASSETS, M.RETURN, M.RISK}


def test_soft_risk_solve(solver, builder):
    """Solve a small SoftRisk instance and validate it produces a feasible budget.

    Sets a target volatility, a generous risk cap, and a risk-priority weight,
    then checks the solved weights are long-only and sum to one.
    """
    builder.parameter[P.SIGMA_TARGET].value = 0.1
    builder.parameter[P.SIGMA_MAX].value = 1.0
    builder.parameter[P.OMEGA].value = 5.0

    problem = builder.build()

    problem.update(
        **{
            D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.MU: np.array([0.25, 0.30]),
            D.MU_UNCERTAINTY: np.zeros(2),
            D.VOLA_UNCERTAINTY: np.zeros(2),
        }
    )

    problem.solve(solver=solver)

    weights = problem.weights[:2]
    assert np.all(weights >= -1e-6)
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=4)
