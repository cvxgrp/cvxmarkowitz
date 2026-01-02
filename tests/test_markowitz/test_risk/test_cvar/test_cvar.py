"""Tests for the CVaR risk model used within portfolio builders.

Validates integration with the MinVar builder and checks solve values for
randomized input data.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.portfolios.min_var import MinVar
from cvxmarkowitz.risk import CVar


def test_estimate_risk(solver):
    """Smoke-test CVaR integration and objective values across updates."""
    model = CVar(alpha=0.95, rows=50, assets=14)

    np.random.seed(42)

    # define the problem
    builder = MinVar(assets=14)

    # overwrite the risk model
    builder.model[M.RISK] = model

    assert M.BOUND_ASSETS in builder.model

    problem = builder.build()

    problem.update(
        **{
            D.RETURNS: np.random.randn(50, 10),
            D.LOWER_BOUND_ASSETS: np.zeros(10),
            D.UPPER_BOUND_ASSETS: np.ones(10),
        }
    )

    # problem = builder.build()
    problem.solve(solver=solver)
    assert problem.value == pytest.approx(0.50587206, abs=1e-5)

    problem.update(
        **{
            D.RETURNS: np.random.randn(50, 10),
            D.LOWER_BOUND_ASSETS: np.zeros(10),
            D.UPPER_BOUND_ASSETS: np.ones(10),
        }
    )

    problem.solve(solver=solver)
    assert problem.value == pytest.approx(0.4355917, abs=1e-5)
