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

from cvx.markowitz.model import Model
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.types import Matrix, Variables
from cvx.markowitz.utils.fill import fill_matrix


@dataclass(frozen=True)
class CVar(Model):
    """Conditional Value-at-Risk (CVaR) model for tail risk estimation."""

    alpha: float = 0.95
    rows: int = 0

    def __post_init__(self) -> None:
        """Initialize the returns parameter matrix."""
        # self.k = int(self.n * (1 - self.alpha))
        self.data[D.RETURNS] = cp.Parameter(
            shape=(self.rows, self.assets),
            name=D.RETURNS,
            value=np.zeros((self.rows, self.assets)),
        )

    def estimate(self, variables: Variables) -> cp.Expression:
        """Estimate CVaR risk as the average of the worst (1-alpha) losses.

        Args:
            variables: Dictionary containing optimization variables with weights.

        Returns:
            CVXPY expression for the CVaR risk estimate.
        """
        # R is a matrix of returns, n is the number of rows in R
        # n = self.R.shape[0]
        # k is the number of returns in the left tail
        # k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        k = int(self.rows * (1 - self.alpha))
        return -cp.sum_smallest(self.data[D.RETURNS] @ variables[D.WEIGHTS], k=k) / k

    def update(self, **kwargs: Matrix) -> None:
        """Update the returns matrix used by the CVaR model.

        Args:
            **kwargs: Must contain 'returns' key with matrix of shape (rows, assets).
        """
        self.data[D.RETURNS].value = fill_matrix(rows=self.rows, cols=self.assets, x=kwargs[D.RETURNS])
