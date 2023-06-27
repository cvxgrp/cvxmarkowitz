# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

import cvxpy as cp

from cvx.markowitz import Model


class CvxError(Exception):
    pass


@dataclass(frozen=True)
class Builder:
    assets: int = 0
    factors: int = None
    model: Dict[str, Model] = field(default_factory=dict)
    constraints: Dict[str, cp.Constraint] = field(default_factory=dict)
    variables: Dict[str, cp.Variable] = field(default_factory=dict)

    def update(self, **kwargs):
        """
        Update the model
        """
        for name, model in self.model.items():
            for key in model.data.keys():
                if key not in kwargs:
                    raise CvxError(f"Missing data for {key} in model {name}")

            model.update(**kwargs)

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

        return cp.Problem(self.objective, list(self.constraints.values()))

    def solution(self, names):
        """
        Return the solution as a dictionary
        """
        return dict(zip(names, self.variables["weights"].value[: len(names)]))

    @property
    def data(self):
        for name, model in self.model.items():
            for key, value in model.data.items():
                yield (name, key), value
