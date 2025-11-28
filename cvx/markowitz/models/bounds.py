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
    """Lower/upper bound model applied to a variable vector.

    Attributes:
        name: Suffix used to distinguish multiple bounds (e.g., "assets").
        acting_on: Key in the variables dict this bound constrains (e.g., D.WEIGHTS).
    """

    name: str = ""
    acting_on: str = "weights"

    def estimate(self, variables: Variables) -> cp.Expression:
        """No estimation for bounds.

        Bounds only contribute constraints; they do not produce an objective term.
        """
        raise NotImplementedError("No estimation for bounds")

    def _f(self, string: str) -> str:
        return f"{string}_{self.name}"

    def __post_init__(self) -> None:
        """Create lower/upper bound parameters with default values.

        Initializes two parameters named with the bound type and `name` suffix,
        both sized to `assets`. Defaults are zeros for lower and ones for upper.
        """
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
        """Assign lower/upper vectors, padding or trimming to asset length."""
        self.data[self._f("lower")].value = fill_vector(num=self.assets, x=kwargs[self._f("lower")])
        self.data[self._f("upper")].value = fill_vector(num=self.assets, x=kwargs[self._f("upper")])

    def constraints(self, variables: Variables) -> Expressions:
        """Return lower/upper inequality constraints for `acting_on` variable.

        Raises KeyError if `acting_on` is not present in `variables`.
        """
        return {
            f"lower bound {self.name}": variables[self.acting_on] >= self.data[self._f("lower")],
            f"upper bound {self.name}": variables[self.acting_on] <= self.data[self._f("upper")],
        }
