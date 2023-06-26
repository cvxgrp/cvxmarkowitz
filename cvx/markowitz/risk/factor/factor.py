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

    def estimate(self, variables):
        """
        Compute the total variance
        """
        # for name, variable in variables.items():
        #    print(name)
        #    print(type(variable))
        #    print(variable.value)

        var_residual = self.residual_risk(variables)
        var_systematic = self.systematic_risk(variables)

        # print(var_residual.value)
        # print(var_systematic.value)
        # print(self.data["chol"].value)
        # print((self.data["chol"] @ self.variables["factor_weights"]).value)

        return cp.norm2(cp.vstack([var_systematic, var_residual]))

    def residual_risk(self, variables):
        return cp.norm2(
            cp.multiply(self.data["idiosyncratic_risk"], variables["weights"])
        )

    def systematic_risk(self, variables):
        return cp.norm2(self.data["chol"] @ variables["factor_weights"])

    def update(self, **kwargs):
        exposure = kwargs["exposure"]
        k, assets = exposure.shape

        self.data["exposure"].value = np.zeros((self.factors, self.assets))
        self.data["exposure"].value[:k, :assets] = kwargs["exposure"]
        self.data["idiosyncratic_risk"].value = np.zeros(self.assets)
        self.data["idiosyncratic_risk"].value[:assets] = kwargs["idiosyncratic_risk"]
        self.data["chol"].value = np.zeros((self.factors, self.factors))
        self.data["chol"].value[:k, :k] = cholesky(kwargs["cov"])

    def constraints(self, variables):
        # factor_weights = kwargs.get("factor_weights", self.data["exposure"] @ weights)
        return {
            "factors": variables["factor_weights"]
            == self.data["exposure"] @ variables["weights"]
        }

    @property
    def variables(self):
        return {
            "weights": cp.Variable(self.assets),
            "factor_weights": cp.Variable(self.factors),
        }

        # return cp.Variable(self.assets), cp.Variable(self.factors)

    def factor_weights(self, variables):
        return self.data["exposure"] @ variables["weights"]
