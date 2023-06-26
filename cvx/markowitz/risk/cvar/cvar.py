# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model


@dataclass(frozen=True)
class CVar(Model):
    """Conditional value at risk model"""

    alpha: float = 0.95
    n: int = 0

    def __post_init__(self):
        # self.k = int(self.n * (1 - self.alpha))
        self.data["R"] = cp.Parameter(
            shape=(self.n, self.assets),
            name="returns",
            value=np.zeros((self.n, self.assets)),
        )

    def estimate(self, variables):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        # R is a matrix of returns, n is the number of rows in R
        # n = self.R.shape[0]
        # k is the number of returns in the left tail
        # k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        k = int(self.n * (1 - self.alpha))
        return -cp.sum_smallest(self.data["R"] @ variables["weights"], k=k) / k

    def update(self, **kwargs):
        ret = kwargs["returns"]
        m = ret.shape[1]

        self.data["R"].value = np.zeros((self.n, self.assets))
        self.data["R"].value[:, :m] = kwargs["returns"]

    def constraints(self, variables):
        return dict({})

    def variables(self):
        return {"weights": cp.Variable(self.assets)}
