# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from cvx.markowitz.risk import SampleCovariance
from cvx.markowitz.solver import Solver


@pytest.fixture()
def solver():
    _solver = Solver(assets=["a", "b", "c"])
    _solver.objective = cp.Maximize(_solver.expected_return([1, 2, 3]))
    _solver.constraints["Funding"] = _solver.funding == 1.0
    _solver.constraints["Long Only"] = _solver.weights >= 0
    return _solver


def test_solver_no_risk(solver):
    x = solver.solve()
    pd.testing.assert_series_equal(
        x, pd.Series(index=["a", "b", "c"], data=np.array([0, 0, 1.0]))
    )


def test_solver_with_risk(solver):
    solver.risk_model = SampleCovariance(assets=3)
    solver.constraints["Risk"] = solver.risk <= 0.9
    print(solver.constraints["Risk"])

    solver.update(
        cov=np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]),
        lower_assets=np.array([0, 0, 0]),
        upper_assets=np.array([1, 1, 1]),
    )

    # todo: constraint is correctly updated, no need to set again
    print(solver.constraints["Risk"])

    x = solver.solve()
    pd.testing.assert_series_equal(
        x,
        pd.Series(
            index=["a", "b", "c"],
            data=np.array([0, 0.2550510227776129, 0.7449489746327344]),
        ),
    )


# todo: Use soft constraint on risk
# todo: Prepare for infeasible problems, define Error class
# todo: Holding costs and Trading costs
