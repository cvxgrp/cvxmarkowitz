# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

import cvxpy as cp

from cvx.markowitz import Model


@dataclass
class Solver:
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
            self.model[name].update(**kwargs)

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
        for name, model in self.model.items():
            self.constraints |= self.model[name].constraints(self.variables)

        return cp.Problem(self.objective, list(self.constraints.values()))

    def solution(self, names):
        """
        Return the solution as a dictionary
        """
        return dict(zip(names, self.variables["weights"].value[: len(names)]))
