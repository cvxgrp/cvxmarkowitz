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
"""Abstract cp model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cvxpy as cp

from cvx.markowitz.types import Expressions, Matrix, Parameter, Variables


@dataclass(frozen=True)
class Model(ABC):
    """Abstract risk model."""

    assets: int
    parameter: Parameter = field(default_factory=dict)
    data: Parameter = field(default_factory=dict)

    @abstractmethod
    def estimate(self, variables: Variables) -> cp.Expression:
        """Estimate the risk metric given the portfolio weights.

        Args:
            variables: Dictionary containing optimization variables.

        Returns:
            CVXPY expression representing the risk estimate.
        """

    @abstractmethod
    def update(self, **kwargs: Matrix) -> None:
        """Update the model parameters with new data.

        Args:
            **kwargs: Keyword arguments containing model data.
        """

    def constraints(self, variables: Variables) -> Expressions:
        """Return constraints for the risk model.

        Args:
            variables: Dictionary containing optimization variables.

        Returns:
            Dictionary of constraint name to constraint expression mappings.
        """
        return {}
