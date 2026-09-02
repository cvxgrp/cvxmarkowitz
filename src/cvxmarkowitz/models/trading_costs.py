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
"""Model for trading costs."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvxmarkowitz.model import Model
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ParameterName as P
from cvxmarkowitz.types import Dimensions, Matrix, Variables
from cvxmarkowitz.utils.fill import fill_vector


@dataclass(frozen=True)
class TradingCosts(Model):
    """Model for trading costs."""

    def __post_init__(self) -> None:
        """Initialize trading cost parameters and previous-weights cache."""
        self.parameter[P.POWER] = cp.Parameter(shape=(), name=P.POWER, value=1.0)

        # initial weights before rebalancing -- keyed by D.WEIGHTS, the same name
        # the decision variable uses, since it is the previous value of it.
        self.data[D.WEIGHTS] = cp.Parameter(shape=self.assets, name=D.WEIGHTS, value=np.zeros(self.assets))

    def estimate(self, variables: Variables) -> cp.Expression:
        """Estimate trading costs for a rebalance.

        Args:
            variables: Optimization variables, expected to contain D.WEIGHTS.

        Returns:
            A convex expression representing the p-power cost of trades
            between current and previous weights.
        """
        return cp.sum(
            cp.power(
                cp.abs(variables[D.WEIGHTS] - self.data[D.WEIGHTS]),
                p=self.parameter[P.POWER],
            )
        )

    def dimensions(self, **kwargs: Matrix) -> Dimensions:
        """Return the number of assets the previous weights imply."""
        return ((D.WEIGHTS, len(kwargs[D.WEIGHTS])),)

    def update(self, **kwargs: Matrix) -> None:
        """Update cached data values.

        Expected keyword arguments:
            weights: Vector of previous weights used as the trading baseline.
        """
        self.data[D.WEIGHTS].value = fill_vector(num=self.assets, x=kwargs[D.WEIGHTS])
