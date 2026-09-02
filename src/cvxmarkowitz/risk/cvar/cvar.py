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

from cvxmarkowitz.cvxerror import CvxDataError
from cvxmarkowitz.model import Model
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.types import Dimensions, Matrix, Variables
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

        Raises:
            CvxDataError: If `alpha` and `rows` leave fewer than one scenario in
                the left tail. Checked here rather than in `estimate` because
                both fields are frozen once construction returns, so the caller
                is told at the point where the mistake can still be corrected.
        """
        if self._tail_size < 1:
            raise CvxDataError(  # noqa: TRY003
                f"alpha={self.alpha} leaves no scenarios in the left tail of rows={self.rows}. "
                f"Lower alpha or raise rows so that int(rows * (1 - alpha)) is at least 1."
            )

        self.data[D.RETURNS] = cp.Parameter(
            shape=(self.rows, self.assets),
            name=D.RETURNS,
            value=np.zeros((self.rows, self.assets)),
        )

    @property
    def _tail_size(self) -> int:
        """Return the number of scenarios averaged over the left tail."""
        return int(self.rows * (1 - self.alpha))

    def estimate(self, variables: Variables) -> cp.Expression:
        """Estimate the risk by computing the Cholesky decomposition of self.cov."""
        # R is a matrix of returns, n is the number of rows in R.
        # k is the number of returns in the left tail; __post_init__ has already
        # rejected any (alpha, rows) pair that would make it zero, which would
        # otherwise reach cvxpy as a bare ValueError and divide by zero here.
        k = self._tail_size
        # average value of the k elements in the left tail
        return -cp.sum_smallest(self.data[D.RETURNS] @ variables[D.WEIGHTS], k=k) / k

    def dimensions(self, **kwargs: Matrix) -> Dimensions:
        """Return the number of assets the scenario matrix implies.

        Its row count is the number of scenarios, which is this model's own
        business rather than a size shared with the other models, so it is not
        declared here.
        """
        return ((D.WEIGHTS, np.shape(kwargs[D.RETURNS])[1]),)

    def update(self, **kwargs: Matrix) -> None:
        """Update the returns matrix used by the CVaR model.

        Expected keyword arguments:
            D.RETURNS: Matrix of historical/scenario returns with shape (rows, assets).
        """
        self.data[D.RETURNS].value = fill_matrix(rows=self.rows, cols=self.assets, x=kwargs[D.RETURNS])
