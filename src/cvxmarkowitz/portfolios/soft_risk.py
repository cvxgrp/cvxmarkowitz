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
"""Portfolio builder with soft risk penalty allowing target risk relaxation."""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp

from cvxmarkowitz.builder import Builder
from cvxmarkowitz.model import Model  # noqa: F401
from cvxmarkowitz.models.expected_returns import ExpectedReturns
from cvxmarkowitz.names import ConstraintName as C
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.names import ParameterName as P
from cvxmarkowitz.types import Parameter, Variables  # noqa: F401


@dataclass(frozen=True)
class SoftRisk(Builder):
    """Maximize w^T mu minus a soft penalty on excess risk.

    The objective is maximize w^T mu - omega * (sigma - sigma_target)_+
    subject to long-only, budget, and max-risk constraints.
    """

    _sigma: cp.Variable = field(default_factory=lambda: cp.Variable(nonneg=True, name="sigma"))
    _sigma_target_times_omega: cp.CallbackParam = field(
        default_factory=lambda: cp.CallbackParam(
            callback=lambda p, q: 0.0, nonneg=True, name="sigma_target_times_omega"
        )
    )

    @property
    def objective(self) -> cp.Maximize:
        """Return the CVXPY objective for soft-risk maximization."""
        expected_return = self.model[M.RETURN].estimate(self.variables)
        soft_risk = cp.pos(self.parameter[P.OMEGA] * self._sigma - self._sigma_target_times_omega)
        return cp.Maximize(expected_return - soft_risk)

    def __post_init__(self) -> None:
        """Initialize models, parameters, and constraints for soft-risk portfolio."""
        super().__post_init__()

        self.model[M.RETURN] = ExpectedReturns(assets=self.assets)

        self.parameter[P.SIGMA_MAX] = cp.Parameter(nonneg=True, name="limit volatility")

        self.parameter[P.SIGMA_TARGET] = cp.Parameter(nonneg=True, name="target volatility")

        self.parameter[P.OMEGA] = cp.Parameter(nonneg=True, name="risk priority")
        self._sigma_target_times_omega._callback = lambda: (
            self.parameter[P.SIGMA_TARGET].value * self.parameter[P.OMEGA].value  # type: ignore[operator]
        )

        self.constraints[C.LONG_ONLY] = self.weights >= 0
        self.constraints[C.BUDGET] = cp.sum(self.weights) == 1.0
        self.constraints[C.RISK] = self.risk.estimate(self.variables) <= self._sigma
        self.constraints["max_risk"] = self._sigma <= self.parameter[P.SIGMA_MAX]
