from __future__ import annotations

import os

import cvxpy as cp
import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.names import ConstraintName as C
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import ModelName as M
from cvx.markowitz.portfolios.min_var import MinVar


@pytest.fixture()
def builder():
    return MinVar(assets=4)


@pytest.fixture()
def problem(builder):
    return builder.build()


def test_models_builder(builder):
    assert builder.model.keys() == {M.BOUND_ASSETS, M.RISK}


def test_constraints(builder):
    assert set(builder.constraints.keys()) == {C.BUDGET, C.LONG_ONLY}


def test_factor_weights(builder):
    with pytest.raises(KeyError):
        builder.factor_weights


def test_weights(builder):
    assert builder.weights.shape == (4,)


def test_is_dpp(problem):
    assert problem.is_dpp()


def test_models_problem(problem):
    assert problem.model.keys() == {M.BOUND_ASSETS, M.RISK}


def test_parameters(problem):
    assert problem.parameter.keys() == {
        D.CHOLESKY,
        D.LOWER_BOUND_ASSETS,
        D.UPPER_BOUND_ASSETS,
        D.VOLA_UNCERTAINTY,
    }


def test_variables(problem):
    assert problem.variables.keys() == {D.WEIGHTS, D._ABS}


@pytest.mark.parametrize("solver", [cp.ECOS, cp.MOSEK, cp.CLARABEL])
def test_min_var(solver):
    if os.getenv("CI", False) and solver == cp.MOSEK:
        pytest.skip("Skipping MOSEK test on CI")

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

    np.testing.assert_almost_equal(
        problem.weights, np.array([0.75, 0.25, 0.0, 0.0]), decimal=3
    )

    assert objective == pytest.approx(0.9354143, abs=1e-5)


@pytest.mark.parametrize("solver", [cp.ECOS, cp.MOSEK, cp.CLARABEL])
def test_min_var_robust(solver):
    if os.getenv("CI", False) and solver == cp.MOSEK:
        pytest.skip("Skipping MOSEK test on CI")

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
