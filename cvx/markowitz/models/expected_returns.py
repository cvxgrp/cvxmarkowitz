"""Model for expected returns"""
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
    """Model for expected returns"""

    def __post_init__(self) -> None:
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
        return self.data[D.MU] @ variables[D.WEIGHTS] - self.parameter[
            "mu_uncertainty"
        ] @ cp.abs(variables[D.WEIGHTS])

    def update(self, **kwargs: Matrix) -> None:
        exp_returns = kwargs[D.MU]
        self.data[D.MU].value = fill_vector(num=self.assets, x=exp_returns)

        # Robust return estimate
        uncertainty = kwargs["mu_uncertainty"]
        if not uncertainty.shape[0] == exp_returns.shape[0]:
            raise CvxError("Mismatch in length for mu and mu_uncertainty")

        self.parameter["mu_uncertainty"].value = fill_vector(
            num=self.assets, x=uncertainty
        )
