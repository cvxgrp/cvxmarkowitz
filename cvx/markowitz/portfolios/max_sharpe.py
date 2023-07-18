# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from cvx.markowitz.builder import Builder
from cvx.markowitz.model import ConstraintName
from cvx.markowitz.models.expected_returns import ExpectedReturns

C = ConstraintName


@dataclass(frozen=True)
class MaxSharpe(Builder):

    """
    Minimize the standard deviation of the portfolio returns subject to a set of constraints
    min StdDev(r_p)
    s.t. w_p >= 0 and sum(w_p) = 1
    """

    @property
    def objective(self):
        return cp.Maximize(self.model["return"].estimate(self.variables))

    def __post_init__(self):
        super().__post_init__()

        self.model["return"] = ExpectedReturns(assets=self.assets)

        self.parameter["sigma_max"] = cp.Parameter(
            nonneg=True, name="maximal volatility"
        )

        self.constraints[C.LONG_ONLY] = self.variables["weights"] >= 0
        self.constraints[C.BUDGET] = cp.sum(self.variables["weights"]) == 1.0
        self.constraints[C.RISK] = (
            self.model["risk"].estimate(self.variables) <= self.parameter["sigma_max"]
        )
