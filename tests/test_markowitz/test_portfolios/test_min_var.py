# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.portfolios.min_var import MinVar


@pytest.mark.parametrize("solver", [cp.ECOS, cp.SCS])
def test_min_var(solver):
    # define the problem

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

    problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.solution(),
        np.array([0.75, 0.25, 0.0, 0.0]),
        decimal=3
        # builder.variables["weights"].value, np.array([0.75, 0.25, 0.0, 0.0]), decimal=5
    )


@pytest.mark.parametrize("solver", [cp.ECOS, cp.SCS, cp.ECOS_BB])
def test_min_var_robust(solver):
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

    problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.solution(),
        np.array([0.626406, 0.373594, 0.0, 0.0]),  # Computed analytically
        decimal=5,
    )
