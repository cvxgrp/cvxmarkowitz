# -*- coding: utf-8 -*-
"""Abstract cp model
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import cvxpy as cp


@dataclass(frozen=True)
class Model(ABC):
    """Abstract risk model"""

    assets: int
    parameter: Dict[str, cp.Parameter] = field(default_factory=dict)
    data: Dict[str, cp.Parameter] = field(default_factory=dict)

    @abstractmethod
    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        """
        Estimate the variance given the portfolio weights
        """

    @abstractmethod
    def update(self, **kwargs):
        """
        Update the data in the risk model
        """

    def constraints(
        self, variables: Dict[str, cp.Variable]
    ) -> Dict[str, cp.Expression]:
        """
        Return the constraints for the risk model
        """
        return {}


class ConstraintName(Enum):
    BUDGET = "budget"
    CONCENTRATION = "concentration"
    LONG_ONLY = "long_only"
    LEVERAGE = "leverage"
    RISK = "risk"

    @classmethod
    def required_constraints(cls) -> list[ConstraintName]:
        """
        Return the required constraints
        """
        return [cls.BUDGET]

    @classmethod
    def try_from_string(cls, string: str) -> ConstraintName | str:
        """
        Try to convert a string to a constraint name, otherwise return the string
        """
        try:
            return cls(string.lower())
        except (AttributeError, ValueError):
            return string

    @classmethod
    def validate_constraints(cls, problem_constraints: list[ConstraintName | str]):
        """
        Validate the presence of all required constraints
        """
        problem_constraints = [cls.try_from_string(c) for c in problem_constraints]

        required_constraints = set(cls.required_constraints())
        assert required_constraints <= set(problem_constraints), (
            f"Missing required constraints: "
            f"{required_constraints - set(problem_constraints)}"
        )
