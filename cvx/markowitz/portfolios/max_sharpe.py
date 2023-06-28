# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from cvx.markowitz.builder import Builder
from cvx.markowitz.models.bounds import Bounds
from cvx.markowitz.models.expected_returns import ExpectedReturns
from cvx.markowitz.models.trading_costs import TradingCosts
from cvx.markowitz.risk import FactorModel
from cvx.markowitz.risk import SampleCovariance


@dataclass(frozen=True)
class MaxSharpe(Builder):

    """
    Minimize the standard deviation of the portfolio returns subject to a set of constraints
    min StdDev(r_p)
    s.t. w_p >= 0 and sum(w_p) = 1
    """

    @property
    def objective(self):
        return cp.Maximize(self.model["return"].estimate(self.variables))

    def __post_init__(self):
        # pick the correct risk model
        if self.factors is not None:
            self.model["risk"] = FactorModel(assets=self.assets, factors=self.factors)

            # add variable for factor weights
            self.variables["factor_weights"] = cp.Variable(
                self.factors, name="factor_weights"
            )
            # add bounds for factor weights
            self.model["bounds_factors"] = Bounds(
                assets=self.factors, name="factors", acting_on="factor_weights"
            )
        else:
            self.model["risk"] = SampleCovariance(assets=self.assets)

        # Note that for the SampleCovariance model the factor_weights are None.
        # They are only included for the harmony of the interfaces for both models.
        self.variables["weights"] = cp.Variable(self.assets, name="weights")
        # add bounds on assets
        self.model["bound_assets"] = Bounds(
            assets=self.assets, name="assets", acting_on="weights"
        )
        self.model["return"] = ExpectedReturns(assets=self.assets)

        self.constraints["long-only"] = self.variables["weights"] >= 0
        self.constraints["fully-invested"] = cp.sum(self.variables["weights"]) == 1.0
        self.constraints["risk"] = self.model["risk"].estimate(self.variables) <= 2.0
