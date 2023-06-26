# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp

from cvx.markowitz.bounds import Bounds
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

        # Please note that for the SampleCovariance model
        # the factor_weights are None and only included for the harmony
        # of the interfaces for both models.
        self.weights_assets, self.weights_factor = self.model.variables

        # add bounds on assets
        self.bounds_assets = Bounds(assets=assets, name="assets")

        # All constraints are combined into a single dictionary
        # Here we use classic long-only and fully-invested constraints.
        # We also combine them with the constraints we inherit from the model.
        # For the | operator please use Python 3.9 or higher.
        self.constraints = {
            "long-only": self.weights_assets >= 0,
            "funding": cp.sum(self.weights_assets) == 1.0,
        }

        self.constraints |= self.model.constraints(
            self.weights_assets, factor_weights=self.weights_factor
        )
        self.constraints |= self.bounds_assets.constraints(self.weights_assets)

        # add bounds on factors
        if factors is not None:
            self.bounds_factors = Bounds(assets=factors, name="factors")
            self.constraints |= self.bounds_factors.constraints(self.weights_factor)

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
        self.bounds_assets.update(**kwargs)

        try:
            self.bounds_factors.update(**kwargs)
        except AttributeError:
            pass

    def solution(self, names):
        """
        Return the solution as a dictionary
        """
        return dict(zip(names, self.weights_assets.value[: len(names)]))
