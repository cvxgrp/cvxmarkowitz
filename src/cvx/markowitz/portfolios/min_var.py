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

from cvx.markowitz.builder import Builder
from cvx.markowitz.names import ConstraintName as C


@dataclass(frozen=True)
class MinVar(Builder):
    """Construct a long-only, budget-constrained minimum-variance portfolio."""

    @property
    def objective(self) -> cp.Objective:
        """Return the CVXPY objective for minimizing portfolio risk."""
        return cp.Minimize(self.risk.estimate(self.variables))

    def __post_init__(self) -> None:
        """Set up default constraints for the minimum-variance portfolio."""
        super().__post_init__()
        self.constraints[C.LONG_ONLY] = self.weights >= 0
        self.constraints[C.BUDGET] = cp.sum(self.weights) == 1.0
