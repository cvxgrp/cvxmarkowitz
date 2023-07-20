# -*- coding: utf-8 -*-
"""Model for holding costs"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model
from cvx.markowitz.model import VariableName

V = VariableName


@dataclass(frozen=True)
class HoldingCosts(Model):
    """Model for holding costs"""

    def __post_init__(self):
        self.data["holding_costs"] = cp.Parameter(
            shape=self.assets, name="holding_costs", value=np.zeros(self.assets)
        )

    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        return cp.sum(
            cp.neg(cp.multiply(variables[V.WEIGHTS], self.data["holding_costs"]))
        )

    def _update(self, x):
        z = np.zeros(self.assets)
        z[: len(x)] = x
        return z

    def update(self, **kwargs):
        self.data["holding_costs"].value = self._update(kwargs["holding_costs"])
