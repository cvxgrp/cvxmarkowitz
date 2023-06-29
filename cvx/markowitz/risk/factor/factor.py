# -*- coding: utf-8 -*-
"""Factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

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
            shape=self.assets,
            name="idiosyncratic_risk",
            value=np.zeros(self.assets),
        )

        self.data["chol"] = cp.Parameter(
            shape=(self.factors, self.factors),
            name="chol",
            value=np.zeros((self.factors, self.factors)),
        )

        self.data["systematic_vola_uncertainty"] = cp.Parameter(
            shape=self.factors,
            name="systematic_vola_uncertainty",
            value=np.zeros(self.factors),
            nonneg=True,
        )

        self.data["idiosyncratic_vola_uncertainty"] = cp.Parameter(
            shape=self.assets,
            name="idiosyncratic_vola_uncertainty",
            value=np.zeros(self.assets),
            nonneg=True,
        )

    def estimate(self, variables) -> cp.Expression:
        """
        Compute the total variance
        """
        var_residual = self._residual_risk(variables)
        var_systematic = self._systematic_risk(variables)

        return cp.norm2(cp.vstack([var_systematic, var_residual]))

    # def _robust_risk(self)

    def _residual_risk(self, variables):
        return cp.norm2(
            cp.hstack(
                [
                    cp.multiply(self.data["idiosyncratic_risk"], variables["weights"]),
                    cp.multiply(
                        self.data["idiosyncratic_vola_uncertainty"],
                        variables["weights"],
                    ),
                ]
            )
        )

    def _systematic_risk(self, variables):
        return cp.norm2(
            cp.hstack(
                [
                    self.data["chol"] @ variables["factor_weights"],
                    self.data["systematic_vola_uncertainty"] @ variables["dummy"],
                ]
            )
        )

    def update(self, **kwargs):
        exposure = kwargs["exposure"]
        k, assets = exposure.shape

        self.data["exposure"].value = np.zeros((self.factors, self.assets))
        self.data["exposure"].value[:k, :assets] = kwargs["exposure"]
        self.data["idiosyncratic_risk"].value = np.zeros(self.assets)
        self.data["idiosyncratic_risk"].value[:assets] = kwargs["idiosyncratic_risk"]
        self.data["chol"].value = np.zeros((self.factors, self.factors))
        self.data["chol"].value[:k, :k] = kwargs["chol"]

        # Robust risk
        self.data["systematic_vola_uncertainty"].value = np.zeros(self.factors)
        self.data["idiosyncratic_vola_uncertainty"].value = np.zeros(self.assets)
        self.data["systematic_vola_uncertainty"].value[:k] = kwargs[
            "systematic_vola_uncertainty"
        ][:k]
        self.data["idiosyncratic_vola_uncertainty"].value[:assets] = kwargs[
            "idiosyncratic_vola_uncertainty"
        ][:assets]

    def constraints(self, variables):
        # factor_weights = kwargs.get("factor_weights", self.data["exposure"] @ weights)
        return {
            "factors": variables["factor_weights"]
            == self.data["exposure"] @ variables["weights"],
            "dummy": variables["dummy"]
            >= cp.abs(variables["factor_weights"]),  # Robust risk dummy variable
        }
