# -*- coding: utf-8 -*-
"""Model for holding costs"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz.model import Model
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.types import Types, UpdateData
from cvx.markowitz.utils.aux import fill_vector


@dataclass(frozen=True)
class HoldingCosts(Model):
    """Model for holding costs"""

    def __post_init__(self) -> None:
        self.data[D.HOLDING_COSTS] = cp.Parameter(
            shape=self.assets, name=D.HOLDING_COSTS, value=np.zeros(self.assets)
        )

    def estimate(self, variables: Types.Variables) -> cp.Expression:
        return cp.sum(
            cp.neg(cp.multiply(variables[D.WEIGHTS], self.data[D.HOLDING_COSTS]))
        )

    def update(self, **kwargs: UpdateData) -> None:
        self.data[D.HOLDING_COSTS].value = fill_vector(
            num=self.assets, x=kwargs[D.HOLDING_COSTS]
        )
