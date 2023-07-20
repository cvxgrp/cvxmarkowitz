# -*- coding: utf-8 -*-
"""Model for holding costs"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import VariableName as V


@dataclass(frozen=True)
class HoldingCosts(Model):
    """Model for holding costs"""

    def __post_init__(self):
        self.data[D.HOLDING_COSTS] = cp.Parameter(
            shape=self.assets, name=D.HOLDING_COSTS, value=np.zeros(self.assets)
        )

    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        return cp.sum(
            cp.neg(cp.multiply(variables[V.WEIGHTS], self.data[D.HOLDING_COSTS]))
        )

    def _update(self, x):
        z = np.zeros(self.assets)
        z[: len(x)] = x
        return z

    def update(self, **kwargs):
        self.data[D.HOLDING_COSTS].value = self._update(kwargs[D.HOLDING_COSTS])
