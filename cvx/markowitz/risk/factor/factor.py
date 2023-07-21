# -*- coding: utf-8 -*-
"""Factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model
from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.utils.aux import fill_matrix, fill_vector


@dataclass(frozen=True)
class FactorModel(Model):
    """Factor risk model"""

    factors: int = 0

    def __post_init__(self):
        self.data[D.EXPOSURE] = cp.Parameter(
            shape=(self.factors, self.assets),
            name=D.EXPOSURE,
            value=np.zeros((self.factors, self.assets)),
        )

        self.data[D.IDIOSYNCRATIC_VOLA] = cp.Parameter(
            shape=self.assets,
            name=D.IDIOSYNCRATIC_VOLA,
            value=np.zeros(self.assets),
        )

        self.data[D.CHOLESKY] = cp.Parameter(
            shape=(self.factors, self.factors),
            name=D.CHOLESKY,
            value=np.zeros((self.factors, self.factors)),
        )

        self.data[D.SYSTEMATIC_VOLA_UNCERTAINTY] = cp.Parameter(
            shape=self.factors,
            name=D.SYSTEMATIC_VOLA_UNCERTAINTY,
            value=np.zeros(self.factors),
            nonneg=True,
        )

        self.data[D.IDIOSYNCRATIC_VOLA_UNCERTAINTY] = cp.Parameter(
            shape=self.assets,
            name=D.IDIOSYNCRATIC_VOLA_UNCERTAINTY,
            value=np.zeros(self.assets),
            nonneg=True,
        )

    def estimate(self, variables) -> cp.Expression:
        """
        Compute the total variance
        """
        var_residual = self._residual_risk(variables)
        var_systematic = self._systematic_risk(variables)

        return cp.norm2(cp.vstack([var_systematic, var_residual]))

    def _residual_risk(self, variables):
        return cp.norm2(
            cp.hstack(
                [
                    cp.multiply(self.data[D.IDIOSYNCRATIC_VOLA], variables[D.WEIGHTS]),
                    cp.multiply(
                        self.data[D.IDIOSYNCRATIC_VOLA_UNCERTAINTY],
                        variables[D.WEIGHTS],
                    ),
                ]
            )
        )

    def _systematic_risk(self, variables):
        return cp.norm2(
            cp.hstack(
                [
                    self.data[D.CHOLESKY] @ variables[D.FACTOR_WEIGHTS],
                    self.data[D.SYSTEMATIC_VOLA_UNCERTAINTY] @ variables[D._ABS],
                ]
            )
        )

    def update(self, **kwargs):
        # check the keywords
        for key in self.data.keys():
            if key not in kwargs.keys():
                raise CvxError(f"Missing keyword {key}")

        if (
            not kwargs[D.IDIOSYNCRATIC_VOLA].shape[0]
            == kwargs[D.IDIOSYNCRATIC_VOLA_UNCERTAINTY].shape[0]
        ):
            raise CvxError(
                "Mismatch in length for idiosyncratic_vola and idiosyncratic_vola_uncertainty"
            )

        exposure = kwargs[D.EXPOSURE]
        k, assets = exposure.shape

        if not kwargs[D.IDIOSYNCRATIC_VOLA].shape[0] == assets:
            raise CvxError("Mismatch in length for idiosyncratic_vola and exposure")

        if not kwargs[D.SYSTEMATIC_VOLA_UNCERTAINTY].shape[0] == k:
            raise CvxError(
                "Mismatch in length of systematic_vola_uncertainty and exposure"
            )

        if not kwargs[D.CHOLESKY].shape[0] == k:
            raise CvxError("Mismatch in size of chol and exposure")

        self.data[D.EXPOSURE].value = fill_matrix(
            rows=self.factors, cols=self.assets, x=kwargs["exposure"]
        )
        self.data[D.IDIOSYNCRATIC_VOLA].value = fill_vector(
            num=self.assets, x=kwargs[D.IDIOSYNCRATIC_VOLA]
        )
        self.data[D.CHOLESKY].value = fill_matrix(
            rows=self.factors, cols=self.factors, x=kwargs[D.CHOLESKY]
        )

        # Robust risk
        self.data[D.SYSTEMATIC_VOLA_UNCERTAINTY].value = fill_vector(
            num=self.factors, x=kwargs[D.SYSTEMATIC_VOLA_UNCERTAINTY]
        )
        self.data[D.IDIOSYNCRATIC_VOLA_UNCERTAINTY].value = fill_vector(
            num=self.assets, x=kwargs[D.IDIOSYNCRATIC_VOLA_UNCERTAINTY]
        )

    def constraints(self, variables):
        # factor_weights = kwargs.get("factor_weights", self.data["exposure"] @ weights)
        return {
            "factors": variables[D.FACTOR_WEIGHTS]
            == self.data[D.EXPOSURE] @ variables[D.WEIGHTS],
            "_abs": variables[D._ABS]
            >= cp.abs(variables[D.FACTOR_WEIGHTS]),  # Robust risk dummy variable
        }
