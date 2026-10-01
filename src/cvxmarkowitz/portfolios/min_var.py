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
"""Minimum-variance portfolio builder."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from cvxmarkowitz.builder import Builder
from cvxmarkowitz.names import ConstraintName as C


@dataclass(frozen=True)
class MinVar(Builder):
    """Construct a long-only, budget-constrained minimum-variance portfolio.

    Example:
        >>> import numpy as np
        >>> from cvxmarkowitz.names import DataNames as D
        >>> problem = MinVar(assets=4).build()
        >>> problem.update(
        ...     **{
        ...         D.CHOLESKY: np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])).T,
        ...         D.LOWER_BOUND_ASSETS: np.zeros(2),
        ...         D.UPPER_BOUND_ASSETS: np.ones(2),
        ...         D.VOLA_UNCERTAINTY: np.zeros(2),
        ...     }
        ... )
        >>> round(problem.solve(), 4)
        0.9354
        >>> round(float(problem.weights.sum()), 6)
        1.0
    """

    @property
    def objective(self) -> cp.Minimize:
        """Return the CVXPY objective for minimizing portfolio risk."""
        return cp.Minimize(self.risk.estimate(self.variables))

    def __post_init__(self) -> None:
        """Set up default constraints for the minimum-variance portfolio."""
        super().__post_init__()
        self.constraints[C.LONG_ONLY] = self.weights >= 0
        self.constraints[C.BUDGET] = cp.sum(self.weights) == 1.0
