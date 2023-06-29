# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from cvx.markowitz.builder import Builder
from cvx.markowitz.models.holding_costs import HoldingCosts
from cvx.markowitz.models.trading_costs import TradingCosts


@dataclass(frozen=True)
class MinVar(Builder):

    """
    Minimize the standard deviation of the portfolio returns subject to a set of constraints
    min StdDev(r_p)
    s.t. w_p >= 0 and sum(w_p) = 1
    """

    @property
    def objective(self):
        return cp.Minimize(self.model["risk"].estimate(self.variables))

    def __post_init__(self):
        super().__post_init__()
        self.constraints["long-only"] = self.variables["weights"] >= 0
        self.constraints["fully-invested"] = cp.sum(self.variables["weights"]) == 1.0
        