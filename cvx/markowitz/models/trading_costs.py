# -*- coding: utf-8 -*-
"""Model for trading costs"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model
from cvx.markowitz.names import VariableName as V


@dataclass(frozen=True)
class TradingCosts(Model):
    """Model for trading costs"""

    def __post_init__(self):
        self.parameter["power"] = cp.Parameter(shape=1, name="power", value=np.ones(1))

        self.data["weights"] = cp.Parameter(
            shape=self.assets, name="weights", value=np.zeros(self.assets)
        )

    def estimate(self, variables: Dict[str | V, cp.Variable]) -> cp.Expression:
        return cp.sum(
            cp.power(
                cp.abs(variables[V.WEIGHTS] - self.data["weights"]),
                p=self.parameter["power"],
            )
        )

    def _update(self, x):
        z = np.zeros(self.assets)
        z[: len(x)] = x
        return z

    def update(self, **kwargs):
        self.data["weights"].value = self._update(kwargs["weights"])
