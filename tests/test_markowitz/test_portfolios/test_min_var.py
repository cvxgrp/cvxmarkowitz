# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import cvxpy as cp
import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.model import ModelName
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.portfolios.min_var import MinVar, estimate_dimensions


@pytest.mark.parametrize("solver", [cp.ECOS, cp.MOSEK, cp.CLARABEL])
def test_min_var(solver):
    if os.getenv("CI", False) and solver == cp.MOSEK:
        pytest.skip("Skipping MOSEK test on CI")

    builder = MinVar(assets=4)

    assert ModelName.BOUND_ASSETS in builder.model
    assert ModelName.RISK in builder.model

    problem = builder.build()

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
    builder = MinVar(assets=4)

    assert ModelName.BOUND_ASSETS in builder.model
    assert ModelName.RISK in builder.model

    problem = builder.build()

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


def test_dimensions():
    input_data = {
        "exposure": np.ones((2, 4)),
        "upper_assets": np.ones(4),
        "lower_assets": np.zeros(4),
    }
    assets, factors = estimate_dimensions(input_data)
    assert assets == 4
    assert factors == 2


def test_dimensions_no_exposure():
    input_data = {
        "upper_assets": np.ones(4),
        "lower_assets": np.zeros(4),
    }
    assets, factors = estimate_dimensions(input_data)
    assert assets == 4
    assert factors is None
