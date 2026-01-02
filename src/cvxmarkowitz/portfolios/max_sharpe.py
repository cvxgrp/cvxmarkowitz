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
"""Portfolio builder maximizing expected return subject to risk and basic constraints."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from cvxmarkowitz.builder import Builder
from cvxmarkowitz.model import Model  # noqa: F401
from cvxmarkowitz.models.expected_returns import ExpectedReturns
from cvxmarkowitz.names import ConstraintName as C
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.names import ParameterName as P
from cvxmarkowitz.types import Parameter, Variables  # noqa: F401


@dataclass(frozen=True)
class MaxSharpe(Builder):
    """Maximize expected return under long-only, budget, and risk constraints."""

    @property
    def objective(self) -> cp.Objective:
        """Return the CVXPY objective for maximizing expected return."""
        return cp.Maximize(self.model[M.RETURN].estimate(self.variables))

    def __post_init__(self) -> None:
        """Initialize models, parameters, and constraints for the builder."""
        super().__post_init__()

        self.model[M.RETURN] = ExpectedReturns(assets=self.assets)

        self.parameter[P.SIGMA_MAX] = cp.Parameter(nonneg=True, name="maximal volatility")

        self.constraints[C.LONG_ONLY] = self.weights >= 0
        self.constraints[C.BUDGET] = cp.sum(self.weights) == 1.0
        self.constraints[C.RISK] = self.risk.estimate(self.variables) <= self.parameter[P.SIGMA_MAX]
