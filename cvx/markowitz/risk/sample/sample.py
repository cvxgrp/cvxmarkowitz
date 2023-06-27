# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.linalg import cholesky
from cvx.markowitz import Model


@dataclass(frozen=True)
class SampleCovariance(Model):
    """Risk model based on the Cholesky decomposition of the sample cov matrix"""

    def __post_init__(self):
        self.data["chol"] = cp.Parameter(
            shape=(self.assets, self.assets),
            name="cholesky of covariance",
            value=np.zeros((self.assets, self.assets)),
        )

    def estimate(self, variables):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        return cp.norm2(self.data["chol"] @ variables["weights"])

    def update(self, **kwargs):
        chol = kwargs["chol"]
        rows = chol.shape[0]
        # assert (
        #    rows <= self.assets
        # ), f"Covariance matrix is too large. Has to be smaller than {self.assets}"
        self.data["chol"].value = np.zeros((self.assets, self.assets))
        self.data["chol"].value[:rows, :rows] = chol

    @property
    def variables(self):
        return {"weights": cp.Variable(self.assets)}
