# -*- coding: utf-8 -*-
"""Abstract cp model
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
