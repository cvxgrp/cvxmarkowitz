# -*- coding: utf-8 -*-
"""Model for holding costs"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model


@dataclass(frozen=True)
class HoldingCosts(Model):
    """Model for holding costs"""

    def __post_init__(self):
        self.data["holding_costs"] = cp.Parameter(
            shape=self.assets, name="holding_costs", value=np.zeros(self.assets)
        )

    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        return cp.sum(
            cp.neg(cp.multiply(variables["weights"], self.data["holding_costs"]))
        )

    def update(self, **kwargs):
        costs = kwargs["holding_costs"]
        num = costs.shape[0]
        self.data["holding_costs"].value = np.zeros(self.assets)
        self.data["holding_costs"].value[:num] = costs
