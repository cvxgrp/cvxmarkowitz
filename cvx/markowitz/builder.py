# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

import cvxpy as cp

from cvx.markowitz import Model
from cvx.markowitz.models.bounds import Bounds
from cvx.markowitz.risk import FactorModel
from cvx.markowitz.risk import SampleCovariance


class CvxError(Exception):
    pass


@dataclass(frozen=True)
class _Problem:
    problem: cp.Problem
    model: Dict[str, Model] = field(default_factory=dict)
    constraints: Dict[str, cp.Constraint] = field(default_factory=dict)
    variables: Dict[str, cp.Variable] = field(default_factory=dict)
    parameter: Dict[str, cp.Parameter] = field(default_factory=dict)
    # problem has var_dict and param_dict

    def update(self, **kwargs):
        """
        Update the model
        """
        for name, model in self.model.items():
            for key in model.data.keys():
                if key not in kwargs:
                    raise CvxError(f"Missing data for {key} in model {name}")

            model.update(**kwargs)

        # set the parameters in the problem
        print(self.problem.param_dict.keys())

        for name, model in self.model.items():
            for key in model.data.keys():
                self.problem.param_dict[key].value = model.data[key].value

    def solve(self, **kwargs):
        """
        Solve the problem
        """
        self.problem.solve(**kwargs)

    def solution(self, variable="weights"):
        """
        Return the solution
        """
        return self.problem.var_dict[variable].value

    @property
    def value(self):
        return self.problem.value

    def is_dpp(self):
        return self.problem.is_dpp()


@dataclass(frozen=True)
class Builder:
    assets: int = 0
    factors: int = None
    model: Dict[str, Model] = field(default_factory=dict)
    constraints: Dict[str, cp.Constraint] = field(default_factory=dict)
    variables: Dict[str, cp.Variable] = field(default_factory=dict)
    parameter: Dict[str, cp.Parameter] = field(default_factory=dict)

    def __post_init__(self):
        # pick the correct risk model
        if self.factors is not None:
            self.model["risk"] = FactorModel(assets=self.assets, factors=self.factors)

            # add variable for factor weights
            self.variables["factor_weights"] = cp.Variable(
                self.factors, name="factor_weights"
            )
            # add bounds for factor weights
            self.model["bounds_factors"] = Bounds(
                assets=self.factors, name="factors", acting_on="factor_weights"
            )
        else:
            self.model["risk"] = SampleCovariance(assets=self.assets)

        # Note that for the SampleCovariance model the factor_weights are None.
        # They are only included for the harmony of the interfaces for both models.
        self.variables["weights"] = cp.Variable(self.assets, name="weights")
        # add bounds on assets
        self.model["bound_assets"] = Bounds(
            assets=self.assets, name="assets", acting_on="weights"
        )

    @property
    @abstractmethod
    def objective(self):
        """
        Return the objective function
        """

    def build(self) -> cp.Problem:
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
        return _Problem(
            problem=problem,
            model=self.model,
            constraints=self.constraints,
            variables=self.variables,
            parameter=self.parameter,
        )

    @property
    def data(self):
        for name, model in self.model.items():
            for key, value in model.data.items():
                yield (name, key), value
