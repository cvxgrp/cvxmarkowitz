# -*- coding: utf-8 -*-
"""Model for trading costs"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz.model import Model
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.utils.aux import fill_vector


@dataclass(frozen=True)
class TradingCosts(Model):
    """Model for trading costs"""

    def __post_init__(self):
        self.parameter["power"] = cp.Parameter(shape=1, name="power", value=np.ones(1))

        # intial weights before rebalancing
        self.data["weights"] = cp.Parameter(
            shape=self.assets, name="weights", value=np.zeros(self.assets)
        )

    def estimate(self, variables: Dict[str | D, cp.Variable]) -> cp.Expression:
        return cp.sum(
            cp.power(
                cp.abs(variables[D.WEIGHTS] - self.data["weights"]),
                p=self.parameter["power"],
            )
        )

    def update(self, **kwargs):
        self.data["weights"].value = fill_vector(num=self.assets, x=kwargs["weights"])
