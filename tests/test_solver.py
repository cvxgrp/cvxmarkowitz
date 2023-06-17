# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from cvx.markowitz.solver import Solver


def test_solver():
    solver = Solver(assets=["a", "b", "c"])
    solver.objective = cp.Maximize(solver.weights @ np.array([1, 2, 3]))
    solver.constraints["Funding"] = solver.funding == 1.0
    solver.constraints["Long Only"] = solver.weights >= 0

    x = solver.solve()
    pd.testing.assert_series_equal(
        x, pd.Series(index=["a", "b", "c"], data=np.array([0, 0, 1.0]))
    )
