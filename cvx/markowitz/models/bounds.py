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
"""Bounds."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz.model import Model
from cvx.markowitz.types import Expressions, Matrix, Variables
from cvx.markowitz.utils.fill import fill_vector


@dataclass(frozen=True)
class Bounds(Model):
    """Model for variable bounds constraints."""

    name: str = ""
    acting_on: str = "weights"

    def estimate(self, variables: Variables) -> cp.Expression:
        """Bounds do not provide risk estimation.

        Args:
            variables: Dictionary containing optimization variables.

        Raises:
            NotImplementedError: Always raised as bounds have no estimate.
        """
        raise NotImplementedError("No estimation for bounds")

    def _f(self, string: str) -> str:
        """Format a string with the bounds name suffix.

        Args:
            string: Base string to format.

        Returns:
            Formatted string with name suffix appended.
        """
        return f"{string}_{self.name}"

    def __post_init__(self) -> None:
        """Initialize lower and upper bound parameters."""
        self.data[self._f("lower")] = cp.Parameter(
            shape=self.assets,
            name=self._f("lower"),
            value=np.zeros(self.assets),
        )
        self.data[self._f("upper")] = cp.Parameter(
            shape=self.assets,
            name=self._f("upper"),
            value=np.ones(self.assets),
        )

    def update(self, **kwargs: Matrix) -> None:
        """Update lower and upper bound values.

        Args:
            **kwargs: Must contain keys for lower and upper bounds.
        """
        self.data[self._f("lower")].value = fill_vector(num=self.assets, x=kwargs[self._f("lower")])
        self.data[self._f("upper")].value = fill_vector(num=self.assets, x=kwargs[self._f("upper")])

    def constraints(self, variables: Variables) -> Expressions:
        """Return lower and upper bound constraints.

        Args:
            variables: Dictionary containing optimization variables.

        Returns:
            Dictionary with lower and upper bound constraint expressions.
        """
        return {
            f"lower bound {self.name}": variables[self.acting_on] >= self.data[self._f("lower")],
            f"upper bound {self.name}": variables[self.acting_on] <= self.data[self._f("upper")],
        }
