# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

import cvxpy as cp

from cvx.markowitz import Model


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
        for _, model in self.model.items():
            for key in model.data.keys():
                if key not in kwargs:
                    warnings.warn(f"Missing data for {key}")
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
        for model in self.model.values():
            for name, constraint in model.constraints(self.variables).items():
                assert name not in self.constraints, "Duplicate constraint"
                self.constraints[name] = constraint

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
