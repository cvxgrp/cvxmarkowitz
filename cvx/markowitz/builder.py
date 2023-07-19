# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import cvxpy as cp

from cvx.markowitz import Model
from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.model import ConstraintName, ModelName, VariableName
from cvx.markowitz.models.bounds import Bounds
from cvx.markowitz.risk import FactorModel, SampleCovariance

V = VariableName
M = ModelName
C = ConstraintName


@dataclass(frozen=True)
class _Problem:
    problem: cp.Problem
    model: Dict[str | ModelName, Model] = field(default_factory=dict)
    # problem has var_dict and param_dict

    def update(self, **kwargs):
        """
        Update the problem
        """
        for name, model in self.model.items():
            for key in model.data.keys():
                if key not in kwargs:
                    raise CvxError(f"Missing data for {key} in model {name}")

            # It's tempting to operate without the models at this stage.
            # However, we would give up a lot of convenience. For example,
            # the models can be prepared to deal with data that has not
            # exactly the correct shape.
            model.update(**kwargs)

        for name, model in self.model.items():
            for key in model.data.keys():
                self.problem.param_dict[key].value = model.data[key].value

        return self

    def solve(self, solver=cp.ECOS, **kwargs):
        """
        Solve the problem
        """
        value = self.problem.solve(solver=solver, **kwargs)

        if self.problem.status is not cp.OPTIMAL:
            raise CvxError(f"Problem status is {self.problem.status}")

        return value

    def solution(self, variable: str = "weights"):
        """
        Return the solution
        """
        return self.problem.var_dict[variable].value

    @property
    def value(self):
        return self.problem.value

    def is_dpp(self):
        return self.problem.is_dpp()

    @property
    def data(self):
        for name, model in self.model.items():
            for key, value in model.data.items():
                yield (name, key), value


@dataclass(frozen=True)
class Builder:
    assets: int = 0
    factors: int = None
    model: Dict[str | ModelName, Model] = field(default_factory=dict)
    constraints: Dict[str | ConstraintName, cp.Constraint] = field(default_factory=dict)
    variables: Dict[str | VariableName, cp.Variable] = field(default_factory=dict)
    parameter: Dict[str, cp.Parameter] = field(default_factory=dict)

    def __post_init__(self):
        # pick the correct risk model
        if self.factors is not None:
            self.model[M.RISK] = FactorModel(assets=self.assets, factors=self.factors)

            # add variable for factor weights
            self.variables[V.FACTOR_WEIGHTS] = cp.Variable(
                self.factors, name=V.FACTOR_WEIGHTS.value
            )
            # add bounds for factor weights
            self.model[M.BOUND_FACTORS] = Bounds(
                assets=self.factors, name="factors", acting_on=V.FACTOR_WEIGHTS
            )
            # add variable for absolute factor weights
            self.variables[V._ABS] = cp.Variable(
                self.factors, name=V._ABS.value, nonneg=True
            )

        else:
            self.model[ModelName.RISK] = SampleCovariance(assets=self.assets)
            #
            # add variable for absolute weights
            self.variables[V._ABS] = cp.Variable(self.assets, name="_abs", nonneg=True)

        # Note that for the SampleCovariance model the factor_weights are None.
        # They are only included for the harmony of the interfaces for both models.
        self.variables[V.WEIGHTS] = cp.Variable(self.assets, name=V.WEIGHTS.value)

        # add bounds on assets
        self.model[M.BOUND_ASSETS] = Bounds(
            assets=self.assets, name="assets", acting_on=VariableName.WEIGHTS
        )

    @property
    @abstractmethod
    def objective(self):
        """
        Return the objective function
        """

    def build(self):
        """
        Build the cvxpy problem
        """
        for name_model, model in self.model.items():
            for name_constraint, constraint in model.constraints(
                self.variables
            ).items():
                self.constraints[f"{name_model}_{name_constraint}"] = constraint

        problem = cp.Problem(self.objective, list(self.constraints.values()))
        assert problem.is_dpp(), "Problem is not DPP"

        ConstraintName.validate_constraints(self.constraints.keys())

        return _Problem(problem=problem, model=self.model)
