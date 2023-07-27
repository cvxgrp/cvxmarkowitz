# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from abc import abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import numpy.typing as npt

from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.model import Model
from cvx.markowitz.models.bounds import Bounds
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import ModelName as M
from cvx.markowitz.risk import FactorModel, SampleCovariance


def deserialize(
    problem_file: str | bytes | PathLike[str] | PathLike[bytes] | int,
) -> _Problem:
    with open(problem_file, "rb") as infile:
        return pickle.load(infile)


@dataclass(frozen=True)
class _Problem:
    problem: cp.Problem
    model: Dict[str, Model] = field(default_factory=dict)

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

        return self

    def solve(self, solver=cp.ECOS, **kwargs):
        """
        Solve the problem
        """
        value = self.problem.solve(solver=solver, **kwargs)

        if self.problem.status is not cp.OPTIMAL:
            raise CvxError(f"Problem status is {self.problem.status}")

        return value

    @property
    def value(self):
        return self.problem.value

    def is_dpp(self) -> bool:
        return self.problem.is_dpp()

    @property
    def data(self):
        for name, model in self.model.items():
            for key, value in model.data.items():
                yield (name, key), value

    @property
    def parameter(self) -> Dict[str, cp.Parameter]:
        return self.problem.param_dict

    @property
    def variables(self) -> Dict[str, cp.Variable]:
        return self.problem.var_dict

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        return self.variables[D.WEIGHTS].value

    @property
    def factor_weights(self) -> npt.NDArray[np.float64]:
        return self.variables[D.FACTOR_WEIGHTS].value

    def serialize(
        self, problem_file: str | bytes | PathLike[str] | PathLike[bytes] | int
    ) -> None:
        with open(problem_file, "wb") as outfile:
            pickle.dump(self, outfile)


@dataclass(frozen=True)
class Builder:
    assets: int = 0
    factors: Optional[int] = None
    model: Dict[str, Model] = field(default_factory=dict)
    constraints: Dict[str, cp.Constraint] = field(default_factory=dict)
    variables: Dict[str, cp.Variable] = field(default_factory=dict)
    parameter: Dict[str, cp.Parameter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # pick the correct risk model
        if self.factors is not None:
            self.model[M.RISK] = FactorModel(assets=self.assets, factors=self.factors)

            # add variable for factor weights
            self.variables[D.FACTOR_WEIGHTS] = cp.Variable(
                self.factors, name=D.FACTOR_WEIGHTS
            )
            # add bounds for factor weights
            self.model[M.BOUND_FACTORS] = Bounds(
                assets=self.factors, name="factors", acting_on=D.FACTOR_WEIGHTS
            )
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
        self.model[M.BOUND_ASSETS] = Bounds(
            assets=self.assets, name="assets", acting_on=D.WEIGHTS
        )

    @property
    @abstractmethod
    def objective(self) -> cp.Expression:
        """
        Return the objective function
        """

    def build(self) -> _Problem:
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

        return _Problem(problem=problem, model=self.model)

    @property
    def weights(self) -> cp.Variable:
        return self.variables[D.WEIGHTS]

    @property
    def risk(self) -> Model:
        return self.model[M.RISK]

    @property
    def factor_weights(self) -> cp.Variable:
        return self.variables[D.FACTOR_WEIGHTS]
