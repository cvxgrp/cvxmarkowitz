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
"""Model for expected returns."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.model import Model
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.types import Matrix, Variables
from cvx.markowitz.utils.fill import fill_vector


@dataclass(frozen=True)
class ExpectedReturns(Model):
    """Model for expected returns."""

    def __post_init__(self) -> None:
        """Initialize expected-return parameters and uncertainty bounds."""
        self.data[D.MU] = cp.Parameter(
            shape=self.assets,
            name=D.MU,
            value=np.zeros(self.assets),
        )

        # Robust return estimate
        self.parameter["mu_uncertainty"] = cp.Parameter(
            shape=self.assets,
            name="mu_uncertainty",
            value=np.zeros(self.assets),
            nonneg=True,
        )

    def estimate(self, variables: Variables) -> cp.Expression:
        """Return robust expected return w^T mu - mu_uncertainty^T |w|.

        Args:
            variables: Optimization variables containing D.WEIGHTS.

        Returns:
            A CVXPY expression for the robust expected return.
        """
        return self.data[D.MU] @ variables[D.WEIGHTS] - self.parameter["mu_uncertainty"] @ cp.abs(variables[D.WEIGHTS])

    def update(self, **kwargs: Matrix) -> None:
        """Update expected returns and their uncertainty bounds.

        Args:
            **kwargs: Must contain 'mu' (expected returns) and 'mu_uncertainty'.

        Raises:
            CvxError: If mu and mu_uncertainty have different lengths.
        """
        exp_returns = kwargs[D.MU]
        self.data[D.MU].value = fill_vector(num=self.assets, x=exp_returns)

        # Robust return estimate
        uncertainty = kwargs["mu_uncertainty"]
        if not uncertainty.shape[0] == exp_returns.shape[0]:
            raise CvxError("Mismatch in length for mu and mu_uncertainty")

        self.parameter["mu_uncertainty"].value = fill_vector(num=self.assets, x=uncertainty)
