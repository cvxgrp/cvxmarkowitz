# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model


@dataclass(frozen=True)
class CVar(Model):
    """Conditional value at risk model"""

    alpha: float = 0.95
    rows: int = 0

    def __post_init__(self):
        # self.k = int(self.n * (1 - self.alpha))
        self.data["returns"] = cp.Parameter(
            shape=(self.rows, self.assets),
            name="returns",
            value=np.zeros((self.rows, self.assets)),
        )

    def estimate(self, variables: Dict[str, cp.Variable]):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        # R is a matrix of returns, n is the number of rows in R
        # n = self.R.shape[0]
        # k is the number of returns in the left tail
        # k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        k = int(self.rows * (1 - self.alpha))
        return -cp.sum_smallest(self.data["returns"] @ variables["weights"], k=k) / k

    def update(self, **kwargs):
        ret = kwargs["returns"]
        columns = ret.shape[1]

        self.data["returns"].value = np.zeros((self.rows, self.assets))
        self.data["returns"].value[:, :columns] = ret
