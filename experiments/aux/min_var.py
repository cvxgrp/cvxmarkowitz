# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp

from cvx.markowitz.bounds import Bounds
from cvx.markowitz.models.holding_costs import HoldingCosts
from cvx.markowitz.models.trading_costs import TradingCosts
from cvx.markowitz.risk import FactorModel
from cvx.markowitz.risk import SampleCovariance


class MinVar:
    """
    Minimize the standard deviation of the portfolio returns subject to a set of constraints
    min StdDev(r_p)
    s.t. w_p >= 0 and sum(w_p) = 1
    """

    def __init__(self, assets: int, factors: int = None):
        self.model = dict()

        # pick the correct risk model
        if factors is not None:
            self.model["risk"] = FactorModel(assets=assets, factors=factors)
        else:
            self.model["risk"] = SampleCovariance(assets=assets)

        # Note that for the SampleCovariance model the factor_weights are None.
        # They are only included for the harmony of the interfaces for both models.
        self.variables = self.model["risk"].variables

        # add bounds on assets
        self.model["bound_assets"] = Bounds(
            assets=assets, name="assets", acting_on="weights"
        )
        # self.model["trading_costs"] = TradingCosts(assets=assets, power=1.0)
        # self.model["holding_costs"] = HoldingCosts(assets=assets)

        # All constraints are combined into a single dictionary
        # Here we use classic long-only and fully-invested constraints.
        # We also combine them with the constraints we inherit from the model.
        # For the | operator please use Python 3.9 or higher.
        self.constraints = {
            "long-only": self.variables["weights"] >= 0,
            "funding": cp.sum(self.variables["weights"]) == 1.0,
        }

        # add bounds on factors
        if factors is not None:
            self.model["bounds_factors"] = Bounds(
                assets=factors, name="factors", acting_on="factor_weights"
            )

        # loop through all models to append constraints
        for name, model in self.model.items():
            self.constraints |= self.model[name].constraints(self.variables)

        # Note that the variables need to be handed over to various models.
        # It's therefore better to have the estimate and constraints methods to get them explicitly.
        self.objective = cp.Minimize(
            self.model["risk"].estimate(self.variables)
            # + self.model["trading_costs"].estimate(self.variables)
            # + self.model["holding_costs"].estimate(self.variables)
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
        for name, model in self.model.items():
            self.model[name].update(**kwargs)

    def solution(self, names):
        """
        Return the solution as a dictionary
        """
        return dict(zip(names, self.variables["weights"].value[: len(names)]))
