from __future__ import annotations

import pickle
from abc import abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Generator

import cvxpy as cp
import numpy as np

from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.model import Model
from cvx.markowitz.models.bounds import Bounds
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import ModelName as M
from cvx.markowitz.risk.factor.factor import FactorModel
from cvx.markowitz.risk.sample.sample import SampleCovariance
from cvx.markowitz.types import File, Matrix, Parameter, Variables


def deserialize(
    problem_file: str | bytes | PathLike[str] | PathLike[bytes] | int,
) -> Any:
    with open(problem_file, "rb") as infile:
        return pickle.load(infile)


@dataclass(frozen=True)
class _Problem:
    problem: cp.Problem
    model: dict[str, Model] = field(default_factory=dict)

    def update(self, **kwargs: Matrix) -> _Problem:
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

    def solve(self, solver: str = cp.ECOS, **kwargs: Any) -> float:
        """
        Solve the problem
        """
        value = self.problem.solve(solver=solver, **kwargs)

        if self.problem.status is not cp.OPTIMAL:
            raise CvxError(f"Problem status is {self.problem.status}")

        return float(value)

    @property
    def value(self) -> float:
        return float(self.problem.value)

    def is_dpp(self) -> bool:
        return bool(self.problem.is_dpp())

    @property
    def data(self) -> Generator[tuple[tuple[str, str], Matrix], None, None]:
        for name, model in self.model.items():
            for key, value in model.data.items():
                yield (name, key), value

    @property
    def parameter(self) -> Parameter:
        return dict(self.problem.param_dict.items())

    @property
    def variables(self) -> Variables:
        return dict(self.problem.var_dict.items())

    @property
    def weights(self) -> Matrix:
        return np.array(self.variables[D.WEIGHTS].value)

    @property
    def factor_weights(self) -> Matrix:
        return np.array(self.variables[D.FACTOR_WEIGHTS].value)

    def serialize(self, problem_file: File) -> None:
        with open(problem_file, "wb") as outfile:
            pickle.dump(self, outfile)


@dataclass(frozen=True)
class Builder:
    assets: int = 0
    factors: int | None = None
    model: dict[str, Model] = field(default_factory=dict)
    constraints: dict[str, cp.Constraint] = field(default_factory=dict)
    variables: Variables = field(default_factory=dict)
    parameter: Parameter = field(default_factory=dict)

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
