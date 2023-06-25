# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz import Model


@dataclass
class ExpectedReturns(Model):
    def __post_init__(self):
        self.data["mu"] = cp.Parameter(
            shape=self.assets,
            name="vector of expected returns",
            value=np.zeros(self.assets),
        )

    def estimate(self, weights, **kwargs):
        return self.data["mu"] @ weights

    def update(self, **kwargs):
        mu = kwargs["mu"]
        n = mu.shape[0]
        self.data["mu"].value = np.zeros(self.assets)
        self.data["mu"].value[:n] = mu

    def constraints(self, weights, **kwargs):
        raise NotImplementedError("No constraints for expected returns")
