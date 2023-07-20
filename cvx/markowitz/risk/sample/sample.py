# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model
from cvx.markowitz.cvxerror import CvxError
from cvx.markowitz.model import VariableName
from cvx.markowitz.names import DataNames as D

V = VariableName


@dataclass(frozen=True)
class SampleCovariance(Model):
    """Risk model based on the Cholesky decomposition of the sample cov matrix"""

    def __post_init__(self):
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
    def estimate(self, variables: Dict[str, cp.Variable]) -> cp.Expression:
        """Estimate the risk by computing the Cholesky decomposition of
        self.cov"""

        return cp.norm2(
            cp.hstack(
                [
                    self.data[D.CHOLESKY] @ variables[V.WEIGHTS],
                    self.data[D.VOLA_UNCERTAINTY] @ variables[V._ABS],
                ]
            )
        )

    def update(self, **kwargs):
        if not kwargs[D.CHOLESKY].shape[0] == kwargs[D.VOLA_UNCERTAINTY].shape[0]:
            raise CvxError("Mismatch in length for chol and vola_uncertainty")

        chol = kwargs[D.CHOLESKY]
        rows = chol.shape[0]
        self.data[D.CHOLESKY].value = np.zeros((self.assets, self.assets))
        self.data[D.CHOLESKY].value[:rows, :rows] = chol

        # Robust risk
        self.data[D.VOLA_UNCERTAINTY].value = np.zeros(self.assets)
        self.data[D.VOLA_UNCERTAINTY].value[:rows] = kwargs[D.VOLA_UNCERTAINTY]

    def constraints(self, variables):
        return {
            "dummy": variables[V._ABS]
            >= cp.abs(variables[V.WEIGHTS]),  # Robust risk dummy variable
        }
