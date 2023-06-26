# -*- coding: utf-8 -*-
"""Factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.linalg import cholesky
from cvx.markowitz import Model


@dataclass(frozen=True)
class FactorModel(Model):
    """Factor risk model"""

    factors: int = 0

    def __post_init__(self):
        self.data["exposure"] = cp.Parameter(
            shape=(self.factors, self.assets),
            name="exposure",
            value=np.zeros((self.factors, self.assets)),
        )

        self.data["idiosyncratic_risk"] = cp.Parameter(
            shape=self.assets, name="idiosyncratic risk", value=np.zeros(self.assets)
        )

        self.data["chol"] = cp.Parameter(
            shape=(self.factors, self.factors),
            name="cholesky of covariance",
            value=np.zeros((self.factors, self.factors)),
        )

    def estimate(self, weights, **kwargs):
        """
        Compute the total variance
        """
        var_residual = cp.norm2(cp.multiply(self.data["idiosyncratic_risk"], weights))

        y = kwargs.get("factor_weights", self.factor_weights(weights))

        return cp.norm2(cp.vstack([cp.norm2(self.data["chol"] @ y), var_residual]))

    def update(self, **kwargs):
        exposure = kwargs["exposure"]
        k, assets = exposure.shape

        self.data["exposure"].value = np.zeros((self.factors, self.assets))
        self.data["exposure"].value[:k, :assets] = kwargs["exposure"]
        self.data["idiosyncratic_risk"].value = np.zeros(self.assets)
        self.data["idiosyncratic_risk"].value[:assets] = kwargs["idiosyncratic_risk"]
        self.data["chol"].value = np.zeros((self.factors, self.factors))
        self.data["chol"].value[:k, :k] = cholesky(kwargs["cov"])

    def constraints(self, weights, **kwargs):
        factor_weights = kwargs.get("factor_weights", self.data["exposure"] @ weights)
        return {"factors": factor_weights == self.data["exposure"] @ weights}

    @property
    def variables(self):
        return cp.Variable(self.assets), cp.Variable(self.factors)

    def factor_weights(self, weights):
        return self.data["exposure"] @ weights
