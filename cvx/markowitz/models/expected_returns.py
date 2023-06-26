# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model


@dataclass(frozen=True)
class ExpectedReturns(Model):
    def __post_init__(self):
        self.data["mu"] = cp.Parameter(
            shape=self.assets,
            name="vector of expected returns",
            value=np.zeros(self.assets),
        )

    def estimate(self, variables):
        return self.data["mu"] @ variables["weights"]

    def update(self, **kwargs):
        mu = kwargs["mu"]
        n = mu.shape[0]
        self.data["mu"].value = np.zeros(self.assets)
        self.data["mu"].value[:n] = mu

    def constraints(self, variables):
        return dict({})
