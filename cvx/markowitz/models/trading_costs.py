# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model


@dataclass(frozen=True)
class TradingCosts(Model):
    def __post_init__(self):
        self.parameter["power"] = cp.Parameter(shape=1, name="power", value=np.ones(1))

        self.data["weights"] = cp.Parameter(
            shape=self.assets, name="weights", value=np.zeros(self.assets)
        )

    def estimate(self, weights, **kwargs):
        # x = weights - self.data["weights"]
        # print(cp.abs(x).value)
        # print(cp.sum(cp.power(cp.abs(x), p=self.parameter["power"])).value)
        return cp.sum(
            cp.power(cp.abs(weights - self.data["weights"]), p=self.parameter["power"])
        )

    def update(self, **kwargs):
        w = kwargs["weights"]
        n = w.shape[0]
        self.data["weights"].value = np.zeros(self.assets)
        self.data["weights"].value[:n] = w

    def constraints(self, weights, **kwargs):
        raise NotImplementedError("No constraints for trading costs")
