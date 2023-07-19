# -*- coding: utf-8 -*-
"""Model for expected returns"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model
from cvx.markowitz.builder import CvxError
from cvx.markowitz.model import VariableName

V = VariableName


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
            name="mu_uncertainty",
            value=np.zeros(self.assets),
            nonneg=True,
        )

    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        return self.data["mu"] @ variables[V.WEIGHTS] - self.parameter[
            "mu_uncertainty"
        ] @ cp.abs(variables[V.WEIGHTS])

    def update(self, **kwargs):
        exp_returns = kwargs["mu"]
        num = exp_returns.shape[0]
        self.data["mu"].value = np.zeros(self.assets)
        self.data["mu"].value[:num] = exp_returns

        # Robust return estimate
        uncertainty = kwargs["mu_uncertainty"]
        if not uncertainty.shape[0] == exp_returns.shape[0]:
            raise CvxError("Mismatch in length for mu and mu_uncertainty")

        self.parameter["mu_uncertainty"].value = np.zeros(self.assets)
        self.parameter["mu_uncertainty"].value[:num] = uncertainty
