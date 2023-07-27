# -*- coding: utf-8 -*-
"""Abstract cp model
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cvxpy as cp

from cvx.markowitz.types import Expressions, Matrix, Parameter, Variables


@dataclass(frozen=True)
class Model(ABC):
    """Abstract risk model"""

    assets: int
    parameter: Parameter = field(default_factory=dict)
    data: Parameter = field(default_factory=dict)

    @abstractmethod
    def estimate(self, variables: Variables) -> cp.Expression:
        """
        Estimate the variance given the portfolio weights
        """

    @abstractmethod
    def update(self, **kwargs: Matrix) -> None:
        """
        Update the data in the risk model
        """

    def constraints(self, variables: Variables) -> Expressions:
        """
        Return the constraints for the risk model
        """
        return {}
