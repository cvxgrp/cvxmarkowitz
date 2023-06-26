# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model


@dataclass(frozen=True)
class HoldingCosts(Model):
    def __post_init__(self):
        self.data["holding_costs"] = cp.Parameter(
            shape=self.assets, name="holding_costs", value=np.zeros(self.assets)
        )

    def estimate(self, weights, **kwargs):
        return cp.sum(cp.neg(cp.multiply(weights, self.data["holding_costs"])))

    def update(self, **kwargs):
        h = kwargs["holding_costs"]
        n = h.shape[0]
        self.data["holding_costs"].value = np.zeros(self.assets)
        self.data["holding_costs"].value[:n] = h

    def constraints(self, weights, **kwargs):
        raise NotImplementedError("No constraints for holding costs")
