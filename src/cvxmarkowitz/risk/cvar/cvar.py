#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Conditional Value-at-Risk (CVaR) risk model implementation."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvxmarkowitz.model import Model
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.types import Matrix, Parameter, Variables  # noqa: F401
from cvxmarkowitz.utils.fill import fill_matrix


@dataclass(frozen=True)
class CVar(Model):
    """Conditional value at risk model."""

    alpha: float = 0.95
    rows: int = 0

    def __post_init__(self) -> None:
        """Initialize CVaR model parameters.

        Creates the returns matrix parameter with shape `(rows, assets)` and
        zeros as default value. The `alpha` quantile controls tail size during
        estimation in `estimate`.
        """
        # self.k = int(self.n * (1 - self.alpha))
        self.data[D.RETURNS] = cp.Parameter(
            shape=(self.rows, self.assets),
            name=D.RETURNS,
            value=np.zeros((self.rows, self.assets)),
        )

    def estimate(self, variables: Variables) -> cp.Expression:
        """Estimate the risk by computing the Cholesky decomposition of self.cov."""
        # R is a matrix of returns, n is the number of rows in R
        # n = self.R.shape[0]
        # k is the number of returns in the left tail
        # k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        k = int(self.rows * (1 - self.alpha))
        return -cp.sum_smallest(self.data[D.RETURNS] @ variables[D.WEIGHTS], k=k) / k

    def update(self, **kwargs: Matrix) -> None:
        """Update the returns matrix used by the CVaR model.

        Expected keyword arguments:
            D.RETURNS: Matrix of historical/scenario returns with shape (rows, assets).
        """
        self.data[D.RETURNS].value = fill_matrix(rows=self.rows, cols=self.assets, x=kwargs[D.RETURNS])
