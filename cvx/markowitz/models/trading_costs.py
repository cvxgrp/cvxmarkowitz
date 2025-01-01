#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Model for trading costs"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz.model import Model
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.types import Matrix, Variables
from cvx.markowitz.utils.fill import fill_vector


@dataclass(frozen=True)
class TradingCosts(Model):
    """Model for trading costs"""

    def __post_init__(self) -> None:
        self.parameter["power"] = cp.Parameter(shape=1, name="power", value=np.ones(1))

        # initial weights before rebalancing
        self.data["weights"] = cp.Parameter(
            shape=self.assets, name="weights", value=np.zeros(self.assets)
        )

    def estimate(self, variables: Variables) -> cp.Expression:
        return cp.sum(
            cp.power(
                cp.abs(variables[D.WEIGHTS] - self.data["weights"]),
                p=self.parameter["power"],
            )
        )

    def update(self, **kwargs: Matrix) -> None:
        self.data["weights"].value = fill_vector(num=self.assets, x=kwargs["weights"])
