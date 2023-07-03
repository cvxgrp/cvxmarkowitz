# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.portfolios.max_sharpe import MaxSharpe


@pytest.mark.parametrize("solver", [cp.ECOS, cp.SCS, cp.CLARABEL])
def test_max_sharpe(solver):
    # define the problem
    builder = MaxSharpe(assets=4)
    builder.parameter["sigma_max"].value = 2.0

    assert "bound_assets" in builder.model
    assert "risk" in builder.model
    assert "return" in builder.model

    problem = builder.build()

    problem.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        mu=np.ones(2),
        mu_uncertainty=np.zeros(2),
        vola_uncertainty=np.zeros(2),
    )

    problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.variables["weights"].value,
        # np.array([5.20124e-01, 4.79876e-01, 0.0, 0.0]),
        # np.array([0.514983, 0.485017, 0.0, 0.0]),
        np.array([5.15358e-01, 4.84642e-01, 0.0, 0.0]),
        decimal=3,
    )

    problem.parameter["sigma_max"].value = 3.0
    problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.variables["weights"].value,
        # np.array([5.10084e-01, 4.89916e-01, 0.0, 0.0]),
        # np.array([0.507383,  0.492617, 0.0, 0.0]),
        np.array([5.07770e-01, 4.92230e-01, 0.0, 0.0]),
        decimal=5,
    )
