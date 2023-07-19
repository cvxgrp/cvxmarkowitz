# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from cvx.markowitz.builder import Builder
from cvx.markowitz.model import ConstraintName, ModelName, VariableName

C = ConstraintName
V = VariableName
M = ModelName


def estimate_dimensions(input_data):
    """Estimate the dimensions of the problem from the input data"""
    assets = input_data["lower_assets"].shape[0]
    try:
        factors = input_data["exposure"].shape[0]
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
        return cp.Minimize(self.model[M.RISK].estimate(self.variables))

    def __post_init__(self):
        super().__post_init__()
        self.constraints[C.LONG_ONLY] = self.variables[V.WEIGHTS] >= 0
        self.constraints[C.BUDGET] = cp.sum(self.variables[V.WEIGHTS]) == 1.0
