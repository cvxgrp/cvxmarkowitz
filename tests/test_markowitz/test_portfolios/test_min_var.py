# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import cvxpy as cp
import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.portfolios.min_var import MinVar


@pytest.mark.parametrize("solver", [cp.ECOS, cp.MOSEK, cp.CLARABEL])
def test_min_var(solver):
    if os.getenv("CI", False) and solver == cp.MOSEK:
        pytest.skip("Skipping MOSEK test on CI")

    builder = MinVar(assets=4)

    assert "bound_assets" in builder.model
    assert "risk" in builder.model

    problem = builder.build()

    problem.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.zeros(2),
    )

    objective = problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.solution(),
        np.array([0.75, 0.25, 0.0, 0.0]),
        decimal=3
        # builder.variables["weights"].value, np.array([0.75, 0.25, 0.0, 0.0]), decimal=5
    )

    assert objective == pytest.approx(0.9354143, abs=1e-5)


@pytest.mark.parametrize("solver", [cp.ECOS, cp.MOSEK, cp.CLARABEL])
def test_min_var_robust(solver):
    if os.getenv("CI", False) and solver == cp.MOSEK:
        pytest.skip("Skipping MOSEK test on CI")

    # define the problem

    builder = MinVar(assets=4)

    assert "bound_assets" in builder.model
    assert "risk" in builder.model

    problem = builder.build()

    problem.update(
        chol=cholesky(np.array([[2.0, 0.4], [0.4, 3.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.array([0.15, 0.3]),
    )

    objective = problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.solution(),
        np.array([0.626406, 0.373594, 0.0, 0.0]),  # Computed analytically
        decimal=4,
    )

    # correct...
    assert objective == pytest.approx(1.1971448, abs=1e-5)
