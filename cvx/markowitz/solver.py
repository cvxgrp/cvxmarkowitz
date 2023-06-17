# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def to_vector(assets, dictionary=None, value=0.0):
    if dictionary is None:
        return np.zeros(len(assets))

    return np.array([dictionary.get(asset, value) for asset in assets])


# @dataclass
class Solver:
    def __init__(self, assets):
        self.assets = assets
        self.weights = cp.Variable(shape=(len(self.assets),), name="weights")
        self.constraints = {}
        self.objective = None

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
