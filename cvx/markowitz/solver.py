# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import pandas as pd

from cvx.risk.model import RiskModel


# def to_vector(assets, dictionary=None, value=0.0):
#    if dictionary is None:
#        return np.zeros(len(assets))
#
#    return np.array([dictionary.get(asset, value) for asset in assets])


# @dataclass
class Solver:
    def __init__(self, assets):
        self.assets = assets
        self.weights = cp.Variable(shape=(len(self.assets),), name="weights")
        self.constraints = {}
        self.objective = None
        self._risk_model = None

    @property
    def funding(self):
        return cp.sum(self.weights)

    @property
    def leverage(self):
        return cp.norm(self.weights, 1)

    def build(self):
        return cp.Problem(self.objective, list(self.constraints.values()))

    def solve(self, **kwargs):
        problem = self.build()
        problem.solve(**kwargs)
        return pd.Series(index=self.assets, data=self.weights.value)

    @property
    def risk_model(self):
        return self._risk_model

    @risk_model.setter
    def risk_model(self, value):
        assert isinstance(value, RiskModel)
        self._risk_model = value

    @property
    def risk(self):
        return self.risk_model.estimate(self.weights)

    def expected_return(self, returns):
        return returns @ self.weights

    def update(self, **kwargs):
        self.risk_model.update(**kwargs)
