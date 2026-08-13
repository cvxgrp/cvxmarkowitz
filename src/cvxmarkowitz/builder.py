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
"""Core builder classes to assemble and solve Markowitz problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cvxpy as cp

from cvxmarkowitz.cvxerror import CvxError
from cvxmarkowitz.model import Model
from cvxmarkowitz.models.bounds import Bounds
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.problem import Problem, deserialize
from cvxmarkowitz.risk.factor.factor import FactorModel
from cvxmarkowitz.risk.sample.sample import SampleCovariance
from cvxmarkowitz.types import Parameter, Variables

# Re-exported for backwards compatibility: ``deserialize``/``Problem`` moved to
# cvxmarkowitz.problem, ``CvxError`` lives in cvxmarkowitz.cvxerror.
__all__ = ["Builder", "CvxError", "Problem", "deserialize"]


@dataclass(frozen=True)
class Builder(ABC):
    """Assemble variables, models, and constraints for Markowitz problems.

    Attributes:
        assets: Number of asset weights to optimize.
        factors: Optional number of factors; if provided, a FactorModel is used,
            otherwise a SampleCovariance risk model is configured.
        model: Mapping of model components (e.g., bounds, risk) by name.
        constraints: Mapping of named cvxpy constraints added during build.
        variables: Mapping of problem variables (weights, factor weights, etc.).
        parameter: Mapping of cvxpy Parameters used by the builder/models.
    """

    assets: int = 0
    factors: int | None = None
    model: dict[str, Model] = field(default_factory=dict)
    constraints: dict[str, cp.Constraint] = field(default_factory=dict)
    variables: Variables = field(default_factory=dict)
    parameter: Parameter = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default risk model, variables, and bounds.

        Selects a factor-based or sample-covariance risk model depending on
        `factors`, creates the corresponding variables (weights and, if
        applicable, factor weights and their absolute values), and registers
        per-asset and/or per-factor bound models.
        """
        # pick the correct risk model
        if self.factors is not None:
            self.model[M.RISK] = FactorModel(assets=self.assets, factors=self.factors)

            # add variable for factor weights
            self.variables[D.FACTOR_WEIGHTS] = cp.Variable(self.factors, name=D.FACTOR_WEIGHTS)
            # add bounds for factor weights
            self.model[M.BOUND_FACTORS] = Bounds(assets=self.factors, name="factors", acting_on=D.FACTOR_WEIGHTS)
            # add variable for absolute factor weights
            self.variables[D._ABS] = cp.Variable(self.factors, name=D._ABS, nonneg=True)

        else:
            self.model[M.RISK] = SampleCovariance(assets=self.assets)
            # add variable for absolute weights
            self.variables[D._ABS] = cp.Variable(self.assets, name=D._ABS, nonneg=True)

        # Note that for the SampleCovariance model the factor_weights are None.
        # They are only included for the harmony of the interfaces for both models.
        self.variables[D.WEIGHTS] = cp.Variable(self.assets, name=D.WEIGHTS)

        # add bounds on assets
        self.model[M.BOUND_ASSETS] = Bounds(assets=self.assets, name="assets", acting_on=D.WEIGHTS)

    @property
    @abstractmethod
    def objective(self) -> cp.Minimize | cp.Maximize:
        """Return the objective function."""

    def build(self) -> Problem:
        """Build the cvxpy problem."""
        for name_model, model in self.model.items():
            for name_constraint, constraint in model.constraints(self.variables).items():
                self.constraints[f"{name_model}_{name_constraint}"] = constraint

        problem = cp.Problem(self.objective, list(self.constraints.values()))
        assert problem.is_dpp(), "Problem is not DPP"  # noqa: S101

        return Problem(problem=problem, model=self.model)

    @property
    def weights(self) -> cp.Variable:
        """Return the asset-weight decision variable (`weights`)."""
        return self.variables[D.WEIGHTS]

    @property
    def risk(self) -> Model:
        """Return the configured risk model held under `model[M.RISK]`."""
        return self.model[M.RISK]

    @property
    def factor_weights(self) -> cp.Variable:
        """Return the factor-weight variable.

        Note: Only present when a factor risk model is used; accessing this
        property without factors configured will raise a KeyError.
        """
        return self.variables[D.FACTOR_WEIGHTS]
