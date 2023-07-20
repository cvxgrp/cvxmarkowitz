# -*- coding: utf-8 -*-
# Import necessary libraries
from __future__ import annotations

import os

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.model import ModelName, VariableName
from cvx.markowitz.portfolios.min_var import MinVar
from cvx.markowitz.risk import CVar

M = ModelName
V = VariableName


@pytest.mark.parametrize("solver", [cp.ECOS, cp.MOSEK, cp.CLARABEL])
def test_estimate_risk(solver):
    """Test the estimate() method"""
    if os.getenv("CI", False) and solver == cp.MOSEK:
        pytest.skip("Skipping MOSEK test on CI")

    model = CVar(alpha=0.95, rows=50, assets=14)

    np.random.seed(42)

    # define the problem
    builder = MinVar(assets=14)

    # overwrite the risk model
    builder.model[M.RISK] = model

    assert M.BOUND_ASSETS in builder.model

    problem = builder.build()

    problem.update(
        returns=np.random.randn(50, 10),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )

    # problem = builder.build()
    problem.solve(solver=solver)
    assert problem.value == pytest.approx(0.50587206, abs=1e-5)

    # it's enough to only update the R value...
    problem.update(
        returns=np.random.randn(50, 10),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )
    problem.solve(solver=solver)
    assert problem.value == pytest.approx(0.4355917, abs=1e-5)
