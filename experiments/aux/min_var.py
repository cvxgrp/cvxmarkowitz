# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp

from cvx.markowitz.risk import FactorModel
from cvx.markowitz.risk import SampleCovariance


class MinVar:
    """
    Minimize the standard deviation of the portfolio returns subject to a set of constraints
    min StdDev(r_p)
    s.t. w_p >= 0 and sum(w_p) = 1
    """

    def __init__(self, assets: int, factors: int = None):
        if factors is not None:
            self.model = FactorModel(assets=assets, factors=factors)
        else:
            self.model = SampleCovariance(assets=assets)

        self.weights_assets, self.weights_factor = self.model.variables
        self.constraints = {
            "long-only": self.weights_assets >= 0,
            "funding": cp.sum(self.weights_assets) == 1.0,
        } | self.model.constraints(
            self.weights_assets, factor_weights=self.weights_factor
        )

        # Note that the variables need to be handed over to various models.
        # It's therefore better to have the estimate and constraints methods to get them explicitly.
        self.objective = cp.Minimize(
            self.model.estimate(self.weights_assets, factor_weights=self.weights_factor)
        )

    def build(self):
        """
        Build the cvxpy problem
        """
        return cp.Problem(self.objective, list(self.constraints.values()))

    def update(self, **kwargs):
        """
        Update the model
        """
        self.model.update(**kwargs)

    def solution(self, names):
        """
        Return the solution as a dictionary
        """
        return dict(zip(names, self.weights_assets.value[: len(names)]))
