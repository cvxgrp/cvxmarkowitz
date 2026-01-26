"""Tests for the MinVar portfolio builder and problem.

These tests cover model wiring, constraints, variables, parameters, and
solving instances with and without robust volatility settings.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxmarkowitz.linalg import cholesky
from cvxmarkowitz.names import ConstraintName as C
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.portfolios.min_var import MinVar


@pytest.fixture
def builder():
    """Construct a MinVar builder fixture with 4 assets."""
    return MinVar(assets=4)


@pytest.fixture
def problem(builder):
    """Build a cvxpy problem from the MinVar builder."""
    return builder.build()


def test_models_builder(builder):
    """Builder should contain the expected models (bounds and risk)."""
    assert builder.model.keys() == {M.BOUND_ASSETS, M.RISK}


def test_constraints(builder):
    """Builder should expose the expected constraints."""
    assert set(builder.constraints.keys()) == {C.BUDGET, C.LONG_ONLY}


def test_factor_weights(builder):
    """Accessing factor weights without factor model should raise KeyError."""
    with pytest.raises(KeyError):
        _ = builder.factor_weights


def test_weights(builder):
    """Weights variable should have the correct shape."""
    assert builder.weights.shape == (4,)


def test_is_dpp(problem):
    """Built problem must satisfy DPP rules."""
    assert problem.is_dpp()


def test_models_problem(problem):
    """Problem should carry expected model components (bounds and risk)."""
    assert problem.model.keys() == {M.BOUND_ASSETS, M.RISK}


def test_parameters(problem):
    """Problem should expose the expected parameter names."""
    assert problem.parameter.keys() == {
        D.CHOLESKY,
        D.LOWER_BOUND_ASSETS,
        D.UPPER_BOUND_ASSETS,
        D.VOLA_UNCERTAINTY,
    }


def test_variables(problem):
    """Problem should expose the expected variable names."""
    assert problem.variables.keys() == {D.WEIGHTS, D._ABS}


def test_min_var(solver):
    """Solve a small MinVar instance and validate objective and weights.

    Args:
        solver: Pytest solver fixture to pass to cvxpy.
    """
    problem = MinVar(assets=4).build()

    problem.update(
        **{
            D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.VOLA_UNCERTAINTY: np.zeros(2),
        }
    )

    objective = problem.solve(solver=solver)

    np.testing.assert_almost_equal(problem.value, 0.9354143466222262)

    np.testing.assert_almost_equal(problem.weights, np.array([0.75, 0.25, 0.0, 0.0]), decimal=3)

    assert objective == pytest.approx(0.9354143, abs=1e-5)


def test_min_var_robust(solver):
    """Solve a robust MinVar instance with uncertainty and compare results.

    Args:
        solver: Pytest solver fixture to pass to cvxpy.
    """
    # define the problem
    problem = MinVar(assets=4).build()

    problem.update(
        **{
            D.CHOLESKY: cholesky(np.array([[2.0, 0.4], [0.4, 3.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.VOLA_UNCERTAINTY: np.array([0.15, 0.3]),
        }
    )

    objective = problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.weights,
        np.array([0.626406, 0.373594, 0.0, 0.0]),  # Computed analytically
        decimal=4,
    )

    assert objective == pytest.approx(1.1971448, abs=1e-5)

    problem.update(
        **{
            D.CHOLESKY: cholesky(np.array([[2.0, 0.4], [0.4, 3.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.VOLA_UNCERTAINTY: np.array([0.3, 0.6]),
        }
    )

    problem.solve(solver=solver)

    assert problem.value > objective
