# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from cvx.markowitz.builder import Builder
from cvx.markowitz.model import ConstraintName as C
from cvx.markowitz.names import DataNames as D


def estimate_dimensions(**kwargs):
    """Estimate the dimensions of the problem from the input data"""
    assets = kwargs[D.LOWER_BOUND_ASSETS].shape[0]
    try:
        factors = kwargs[D.EXPOSURE].shape[0]
    except KeyError:
        factors = None

    return assets, factors


@dataclass(frozen=True)
class MinVar(Builder):

    """
    Minimize the standard deviation of the portfolio returns subject to a set of constraints
    min StdDev(r_p)
    s.t. w_p >= 0 and sum(w_p) = 1
    """

    @property
    def objective(self):
        return cp.Minimize(self.risk.estimate(self.variables))

    def __post_init__(self):
        super().__post_init__()
        self.constraints[C.LONG_ONLY] = self.weights >= 0
        self.constraints[C.BUDGET] = cp.sum(self.weights) == 1.0
