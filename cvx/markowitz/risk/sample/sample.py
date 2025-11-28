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
"""Risk models based on the sample covariance matrix."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.model import Model
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.types import Expressions, Matrix, Variables
from cvx.markowitz.utils.fill import fill_matrix, fill_vector


@dataclass(frozen=True)
class SampleCovariance(Model):
    """Risk model based on the Cholesky decomposition of the sample cov matrix."""

    def __post_init__(self) -> None:
        """Initialize Cholesky and volatility uncertainty parameters."""
        self.data[D.CHOLESKY] = cp.Parameter(
            shape=(self.assets, self.assets),
            name=D.CHOLESKY,
            value=np.zeros((self.assets, self.assets)),
        )

        self.data[D.VOLA_UNCERTAINTY] = cp.Parameter(
            shape=self.assets,
            name=D.VOLA_UNCERTAINTY,
            value=np.zeros(self.assets),
            nonneg=True,
        )

    # x: array([ 5.19054e-01,  4.80946e-01, -1.59557e-12, -1.59557e-12])
    def estimate(self, variables: Variables) -> cp.Expression:
        """Estimate risk via Cholesky-based norm of exposures and uncertainties.

        Args:
            variables: Dictionary containing optimization variables.

        Returns:
            CVXPY expression for the risk estimate.
        """
        return cp.norm2(
            cp.hstack(
                [
                    self.data[D.CHOLESKY] @ variables[D.WEIGHTS],
                    self.data[D.VOLA_UNCERTAINTY] @ variables[D._ABS],
                ]
            )
        )

    def update(self, **kwargs: Matrix) -> None:
        """Assign Cholesky factor and volatility-uncertainty vector.

        Args:
            **kwargs: Must contain 'chol' and 'vola_uncertainty' keys.

        Raises:
            CvxError: If chol and vola_uncertainty have mismatched dimensions.
        """
        if not kwargs[D.CHOLESKY].shape[0] == kwargs[D.VOLA_UNCERTAINTY].shape[0]:
            raise CvxError("Mismatch in length for chol and vola_uncertainty")

        self.data[D.CHOLESKY].value = fill_matrix(rows=self.assets, cols=self.assets, x=kwargs[D.CHOLESKY])
        self.data[D.VOLA_UNCERTAINTY].value = fill_vector(num=self.assets, x=kwargs[D.VOLA_UNCERTAINTY])

    def constraints(self, variables: Variables) -> Expressions:
        """Return auxiliary constraints used for robust risk modeling.

        Args:
            variables: Dictionary containing optimization variables.

        Returns:
            Dictionary with absolute value dummy constraint for robust risk.
        """
        return {
            "dummy": variables[D._ABS] >= cp.abs(variables[D.WEIGHTS]),  # Robust risk dummy variable
        }
