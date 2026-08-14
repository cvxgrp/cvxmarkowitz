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

from cvxmarkowitz.cvxerror import CvxBuildError, CvxDataError, CvxError
from cvxmarkowitz.model import Model

# `Bounds` is imported concretely on purpose: it is not a default being chosen
# among alternatives but part of what a Builder unconditionally is, so putting
# it behind a selector would be indirection with nothing to select. The risk
# model *is* a choice, and `default_risk_model` owns it -- see cvxmarkowitz.risk.
from cvxmarkowitz.models.bounds import Bounds
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.problem import Problem
from cvxmarkowitz.risk import default_risk_model
from cvxmarkowitz.types import Parameter, Variables

# Re-exported for backwards compatibility: ``Problem`` moved to
# cvxmarkowitz.problem, ``CvxError`` lives in cvxmarkowitz.cvxerror.
__all__ = ["Builder", "CvxError", "Problem"]


@dataclass(frozen=True)
class Builder(ABC):
    """Assemble variables, models, and constraints for Markowitz problems.

    Attributes:
        assets: Number of asset weights to optimize.
        factors: Optional number of factors; if provided, a FactorModel is used,
            otherwise a SampleCovariance risk model is configured. Ignored for the
            choice of risk model when one is injected via `model`, but still
            controls which variables and bounds are created.
        model: Mapping of model components (e.g., bounds, risk) by name. Pass an
            entry under `ModelName.RISK` to supply your own risk model instead of
            the `factors`-based default -- see `__post_init__`.
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
        """Initialize the risk model, variables, and bounds.

        Creates the variables (weights and, if `factors` is given, factor weights
        and their absolute values) and registers the per-asset and/or per-factor
        bound models.

        The risk model is only defaulted when the caller did not supply one.
        Passing `model={ModelName.RISK: my_model}` to the constructor keeps that
        model, which is how risk models outside the two defaults -- `CVar`, say --
        are used with a builder:

            MinVar(assets=10, model={M.RISK: CVar(assets=10, rows=100)})

        With no entry under `ModelName.RISK`, `cvxmarkowitz.risk.default_risk_model`
        picks one: a `FactorModel` when `factors` is set, a `SampleCovariance`
        otherwise.
        """
        if self.factors is not None:
            # add variable for factor weights
            self.variables[D.FACTOR_WEIGHTS] = cp.Variable(self.factors, name=D.FACTOR_WEIGHTS)
            # add bounds for factor weights
            self.model[M.BOUND_FACTORS] = Bounds(assets=self.factors, name="factors", acting_on=D.FACTOR_WEIGHTS)
            # add variable for absolute factor weights
            self.variables[D._ABS] = cp.Variable(self.factors, name=D._ABS, nonneg=True)

        else:
            # add variable for absolute weights
            self.variables[D._ABS] = cp.Variable(self.assets, name=D._ABS, nonneg=True)

        # pick the default risk model, unless the caller injected one
        if M.RISK not in self.model:
            self.model[M.RISK] = default_risk_model(assets=self.assets, factors=self.factors)

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
        """Build the cvxpy problem.

        Raises:
            CvxBuildError: If the assembled problem is not DPP-compliant. This is
                checked with a raise rather than an `assert` on purpose: `assert`
                is stripped under `python -O`, and DPP compliance is the invariant
                the whole caching story rests on.
        """
        for name_model, model in self.model.items():
            for name_constraint, constraint in model.constraints(self.variables).items():
                self.constraints[f"{name_model}_{name_constraint}"] = constraint

        problem = cp.Problem(self.objective, list(self.constraints.values()))

        if not problem.is_dpp():
            raise CvxBuildError(  # noqa: TRY003
                "The assembled problem is not DPP-compliant, so cvxpy cannot cache "
                "its canonicalization. Check the objective and the constraints for "
                "expressions that are not affine in the parameters."
            )

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

        Raises:
            CvxDataError: If the builder was constructed without `factors`, in
                which case there is no factor-weight variable to return.
        """
        try:
            return self.variables[D.FACTOR_WEIGHTS]
        except KeyError as err:
            raise CvxDataError(  # noqa: TRY003
                "No factor weights: this builder was constructed without 'factors'. "
                "Pass factors=<number of factors> to use a factor risk model."
            ) from err
