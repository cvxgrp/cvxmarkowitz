# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model
from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.model import VariableName

V = VariableName


@dataclass(frozen=True)
class SampleCovariance(Model):
    """Risk model based on the Cholesky decomposition of the sample cov matrix"""

    def __post_init__(self):
        self.data["chol"] = cp.Parameter(
            shape=(self.assets, self.assets),
            name="chol",
            value=np.zeros((self.assets, self.assets)),
        )

        self.data["vola_uncertainty"] = cp.Parameter(
            shape=self.assets,
            name="vola_uncertainty",
            value=np.zeros(self.assets),
            nonneg=True,
        )

    # x: array([ 5.19054e-01,  4.80946e-01, -1.59557e-12, -1.59557e-12])
    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        """Estimate the risk by computing the Cholesky decomposition of
        self.cov"""

        return cp.norm2(
            cp.hstack(
                [
                    self.data["chol"] @ variables[V.WEIGHTS],
                    self.data["vola_uncertainty"] @ variables[V._ABS],
                ]
            )
        )  #

        # return cp.sum_squares(self.data["chol"] @ variables["weights"]) \
        # + (self.data["vola_uncertainty"] @ cp.abs(variables["weights"]))**2  # Robust risk

    def update(self, **kwargs):
        if not kwargs["vola_uncertainty"].shape[0] == kwargs["chol"].shape[0]:
            raise CvxError("Mismatch in length for chol and vola_uncertainty")

        chol = kwargs["chol"]
        rows = chol.shape[0]
        self.data["chol"].value = np.zeros((self.assets, self.assets))
        self.data["chol"].value[:rows, :rows] = chol

        # Robust risk
        self.data["vola_uncertainty"].value = np.zeros(self.assets)
        self.data["vola_uncertainty"].value[:rows] = kwargs["vola_uncertainty"]

    def constraints(self, variables):
        return {
            "dummy": variables[V._ABS]
            >= cp.abs(variables[V.WEIGHTS]),  # Robust risk dummy variable
        }
