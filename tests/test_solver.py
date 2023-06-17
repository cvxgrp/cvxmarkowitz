# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from cvx.markowitz.solver import Solver


class TestSolver(Solver):
    """Max Sharpe solver"""

    def add_constraints(self, **kwargs):
        # self.constraints["Risk"] = self.variance() <= kwargs.get("Risk", 1.0)
        # self.constraints["Lower"] = self.funding * self.lower_bound <= self.weights
        # self.constraints["Upper"] = self.funding * self.upper_bound >= self.weights
        # exclude the zero solution...
        # self.constraints["Investment"] = self.funding >= 0.1

        # self.constraints["Lower"] = self.lower_bound <= self.weights
        # self.constraints["Upper"] = self.upper_bound >= self.weights
        self.constraints["Funding"] = self.funding == 1.0
        self.constraints["Long Only"] = self.weights >= 0

    def objective(self, **kwargs):
        return cp.Maximize(self.weights @ np.array([1, 2, 3]))

    # def objective(self, **kwargs):
    #    return cvx.Maximize(self.expected_return)

    def solve(self, **kwargs):
        return self._solve(scale=True, **kwargs)


def test_solver():
    solver = Solver(assets=["a", "b", "c"])
    solver.objective = cp.Maximize(solver.weights @ np.array([1, 2, 3]))
    solver.constraints["Funding"] = solver.funding == 1.0
    solver.constraints["Long Only"] = solver.weights >= 0

    x = solver.solve()
    pd.testing.assert_series_equal(
        x, pd.Series(index=["a", "b", "c"], data=np.array([0, 0, 1.0]))
    )
