# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import cvxpy as cp
import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.portfolios.max_sharpe import MaxSharpe


@pytest.mark.parametrize("solver", [cp.ECOS, cp.CLARABEL, cp.MOSEK])
def test_max_sharpe(solver):
    if os.getenv("CI", False) and solver == cp.MOSEK:
        pytest.skip("Skipping MOSEK test on CI")

    # define the problem
    builder = MaxSharpe(assets=4)
    builder.parameter["sigma_max"].value = 1.0

    assert "bound_assets" in builder.model
    assert "risk" in builder.model
    assert "return" in builder.model

    problem = builder.build()

    problem.update(
        chol=cholesky(np.array([[1.0, 0.6], [0.6, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        mu=np.array([0.25, 0.30]),
        mu_uncertainty=np.zeros(2),
        vola_uncertainty=np.zeros(2),
    )

    x = problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.variables["weights"].value,
        np.array([5.5556e-01, 4.444e-01, 0.0, 0.0]),
        decimal=4)
