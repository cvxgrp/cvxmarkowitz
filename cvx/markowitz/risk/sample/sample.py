# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.linalg import cholesky
from cvx.markowitz import Model
from cvx.markowitz.bounds import Bounds


@dataclass
class SampleCovariance(Model):
    """Risk model based on the Cholesky decomposition of the sample cov matrix"""

    def __post_init__(self):
        self.data["chol"] = cp.Parameter(
            shape=(self.assets, self.assets),
            name="cholesky of covariance",
            value=np.zeros((self.assets, self.assets)),
        )
        self.bounds = Bounds(assets=self.assets, name="assets")

    def estimate(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        return cp.norm2(self.data["chol"] @ weights)

    def update(self, **kwargs):
        cov = kwargs["cov"]
        n = cov.shape[0]
        self.data["chol"].value = np.zeros((self.assets, self.assets))
        self.data["chol"].value[:n, :n] = cholesky(cov)
        self.bounds.update(**kwargs)

    def constraints(self, weights, **kwargs):
        return self.bounds.constraints(weights)

    @property
    def variables(self):
        return cp.Variable(self.assets), None
