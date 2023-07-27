# -*- coding: utf-8 -*-
"""Abstract cp model
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import cvxpy as cp

from cvx.markowitz.types import Types, UpdateData


@dataclass(frozen=True)
class Model(ABC):
    """Abstract risk model"""

    assets: int
    parameter: Types.Parameter = field(default_factory=dict)
    data: Types.Parameter = field(default_factory=dict)

    @abstractmethod
    def estimate(self, variables: Types.Variables) -> cp.Expression:
        """
        Estimate the variance given the portfolio weights
        """

    @abstractmethod
    def update(self, **kwargs: UpdateData) -> None:
        """
        Update the data in the risk model
        """

    def constraints(self, variables: Types.Variables) -> Dict[str, cp.Expression]:
        """
        Return the constraints for the risk model
        """
        return {}
