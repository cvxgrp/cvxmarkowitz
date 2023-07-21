# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.utils.aux import fill_matrix


@dataclass(frozen=True)
class CVar(Model):
    """Conditional value at risk model"""

    alpha: float = 0.95
    rows: int = 0

    def __post_init__(self):
        # self.k = int(self.n * (1 - self.alpha))
        self.data[D.RETURNS] = cp.Parameter(
            shape=(self.rows, self.assets),
            name=D.RETURNS,
            value=np.zeros((self.rows, self.assets)),
        )

    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        # R is a matrix of returns, n is the number of rows in R
        # n = self.R.shape[0]
        # k is the number of returns in the left tail
        # k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        k = int(self.rows * (1 - self.alpha))
        return -cp.sum_smallest(self.data[D.RETURNS] @ variables[D.WEIGHTS], k=k) / k

    def update(self, **kwargs):
        self.data[D.RETURNS].value = fill_matrix(
            rows=self.rows, cols=self.assets, x=kwargs[D.RETURNS]
        )
