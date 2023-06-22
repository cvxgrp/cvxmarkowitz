# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.linalg import cholesky
from cvx.markowitz import Model
from cvx.markowitz.bounds import Bounds


@dataclass
class SampleCovariance(Model):
    """Risk model based on the Cholesky decomposition of the sample cov matrix"""

    num: int = 0

    def __post_init__(self):
        self.parameter["chol"] = cvx.Parameter(
            shape=(self.num, self.num),
            name="cholesky of covariance",
            value=np.zeros((self.num, self.num)),
        )
        self.bounds = Bounds(m=self.num, name="assets")

    def estimate(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        return cvx.norm2(self.parameter["chol"] @ weights)

    def update(self, **kwargs):
        cov = kwargs["cov"]
        n = cov.shape[0]

        self.parameter["chol"].value[:n, :n] = cholesky(cov)
        self.bounds.update(**kwargs)

    def constraints(self, weights):
        return self.bounds.constraints(weights)
