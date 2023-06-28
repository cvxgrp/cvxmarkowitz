# -*- coding: utf-8 -*-
"""Model for expected returns"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model


@dataclass(frozen=True)
class ExpectedReturns(Model):
    """Model for expected returns"""

    def __post_init__(self):
        self.data["mu"] = cp.Parameter(
            shape=self.assets,
            name="mu",
            value=np.zeros(self.assets),
        )

        # Robust return estimate
        self.parameter["mu_uncertainty"] = cp.Parameter(
            shape=self.assets,
            name="uncertainty in expected returns",
            value=np.zeros(self.assets),
            nonneg=True,
        )

    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        return self.data["mu"] @ variables["weights"] \
            - self.parameter["mu_uncertainty"] @ cp.abs(variables["weights"])

    def update(self, **kwargs):
        exp_returns = kwargs["mu"]
        num = exp_returns.shape[0]
        self.data["mu"].value = np.zeros(self.assets)
        self.data["mu"].value[:num] = exp_returns

        # Robust return estimate
        uncertainty = kwargs["mu_uncertainty"]
        self.parameter["mu_uncertainty"].value = np.zeros(self.assets)
        self.parameter["mu_uncertainty"].value[:num] = uncertainty

