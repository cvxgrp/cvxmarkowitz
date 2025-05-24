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
"""
Portfolio optimization problem builder module.

This module provides classes for building and solving portfolio optimization problems
using convex optimization. It includes a Builder class for constructing problems
and a _Problem class for solving and analyzing the results.

The Builder class is designed to be extended by specific portfolio optimization
strategies (like minimum variance, maximum Sharpe ratio, etc.) by implementing
the abstract objective method.
"""

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
    """
    Deserialize a problem from a file.

    Args:
        problem_file: Path to the file containing the serialized problem.

    Returns:
        The deserialized problem object.
    """
    with open(problem_file, "rb") as infile:
        return pickle.load(infile)


@dataclass(frozen=True)
class _Problem:
    """
    Internal class representing a built optimization problem.

    This class encapsulates a CVXPY problem and its associated models,
    providing methods to update parameters, solve the problem, and
    extract results.

    Attributes:
        problem: The CVXPY problem object.
        model: Dictionary mapping model names to Model objects.
    """

    problem: cp.Problem
    model: dict[str, Model] = field(default_factory=dict)

    def update(self, **kwargs: Matrix) -> _Problem:
        """
        Update the problem with new data.

        This method updates all models in the problem with the provided data.

        Args:
            **kwargs: Dictionary of matrices containing the data to update.
                     Each key should correspond to a data key in one of the models.

        Returns:
            The updated problem instance (self).

        Raises:
            CvxError: If any required data key is missing from kwargs.
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

    def solve(self, solver: str = cp.CLARABEL, **kwargs: Any) -> float:
        """
        Solve the optimization problem.

        Args:
            solver: The CVXPY solver to use (default: CLARABEL).
            **kwargs: Additional keyword arguments to pass to the solver.

        Returns:
            The optimal value of the objective function.

        Raises:
            CvxError: If the problem status is not OPTIMAL after solving.
        """
        value = self.problem.solve(solver=solver, **kwargs)

        if self.problem.status is not cp.OPTIMAL:
            raise CvxError(f"Problem status is {self.problem.status}")

        return float(value)

    @property
    def value(self) -> float:
        """
        Get the optimal value of the objective function.

        Returns:
            The optimal value as a float.
        """
        return float(self.problem.value)

    def is_dpp(self) -> bool:
        """
        Check if the problem is DPP (Disciplined Parametrized Programming) compliant.

        Returns:
            True if the problem is DPP compliant, False otherwise.
        """
        return bool(self.problem.is_dpp())

    @property
    def data(self) -> Generator[tuple[tuple[str, str], Matrix]]:
        """
        Get all data used in the problem's models.

        Returns:
            A generator yielding tuples of ((model_name, data_key), data_value).
        """
        for name, model in self.model.items():
            for key, value in model.data.items():
                yield (name, key), value

    @property
    def parameter(self) -> Parameter:
        """
        Get all parameters in the problem.

        Returns:
            A dictionary mapping parameter names to parameter objects.
        """
        return dict(self.problem.param_dict.items())

    @property
    def variables(self) -> Variables:
        """
        Get all variables in the problem.

        Returns:
            A dictionary mapping variable names to variable objects.
        """
        return dict(self.problem.var_dict.items())

    @property
    def weights(self) -> Matrix:
        """
        Get the optimal portfolio weights.

        Returns:
            A numpy array containing the optimal weights for each asset.
        """
        return np.array(self.variables[D.WEIGHTS].value)

    @property
    def factor_weights(self) -> Matrix:
        """
        Get the optimal factor weights (for factor models).

        Returns:
            A numpy array containing the optimal weights for each factor.
        """
        return np.array(self.variables[D.FACTOR_WEIGHTS].value)

    def serialize(self, problem_file: File) -> None:
        """
        Serialize the problem to a file.

        Args:
            problem_file: Path to the file where the problem will be serialized.
        """
        with open(problem_file, "wb") as outfile:
            pickle.dump(self, outfile)


@dataclass(frozen=True)
class Builder:
    """
    Abstract base class for building portfolio optimization problems.

    This class provides the foundation for constructing portfolio optimization
    problems using either sample covariance or factor models for risk estimation.
    Concrete subclasses must implement the objective method to define the
    specific optimization objective.

    Attributes:
        assets: Number of assets in the portfolio.
        factors: Number of factors for factor models (None for sample covariance).
        model: Dictionary mapping model names to Model objects.
        constraints: Dictionary mapping constraint names to CVXPY constraints.
        variables: Dictionary mapping variable names to CVXPY variables.
        parameter: Dictionary mapping parameter names to CVXPY parameters.
    """

    assets: int = 0
    factors: int | None = None
    model: dict[str, Model] = field(default_factory=dict)
    constraints: dict[str, cp.Constraint] = field(default_factory=dict)
    variables: Variables = field(default_factory=dict)
    parameter: Parameter = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the builder after instance creation.

        This method sets up the appropriate risk model (factor or sample covariance)
        based on whether factors are specified, creates the necessary variables,
        and adds default constraints.
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
    def objective(self) -> cp.Expression:
        """
        Define the objective function for the optimization problem.

        This abstract method must be implemented by concrete subclasses to
        define the specific optimization objective (e.g., minimize risk,
        maximize return, etc.).

        Returns:
            A CVXPY expression representing the objective function.
        """
        pass

    def build(self) -> _Problem:
        """
        Build the complete CVXPY optimization problem.

        This method collects all constraints from the models, creates a CVXPY
        Problem with the objective function, and verifies that the problem
        is DPP (Disciplined Parametrized Programming) compliant.

        Returns:
            A _Problem object encapsulating the built optimization problem.

        Raises:
            AssertionError: If the problem is not DPP compliant.
        """
        for name_model, model in self.model.items():
            for name_constraint, constraint in model.constraints(self.variables).items():
                self.constraints[f"{name_model}_{name_constraint}"] = constraint

        problem = cp.Problem(self.objective, list(self.constraints.values()))
        assert problem.is_dpp(), "Problem is not DPP"

        return _Problem(problem=problem, model=self.model)

    @property
    def weights(self) -> cp.Variable:
        """
        Get the portfolio weights variable.

        Returns:
            The CVXPY variable representing portfolio weights.
        """
        return self.variables[D.WEIGHTS]

    @property
    def risk(self) -> Model:
        """
        Get the risk model.

        Returns:
            The Model object used for risk estimation.
        """
        return self.model[M.RISK]

    @property
    def factor_weights(self) -> cp.Variable:
        """
        Get the factor weights variable (for factor models).

        Returns:
            The CVXPY variable representing factor weights.
        """
        return self.variables[D.FACTOR_WEIGHTS]
