# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from cvx.linalg import pca
from cvx.markowitz.model import Model
from cvx.markowitz.risk import FactorModel


class Solver:
    def __init__(self, assets: int, k=None):
        """
        Initialize the solver
        :param assets: number of assets
        :param k: number of factors
        """
        self.assets = assets
        self.k = k or assets

        self.weights = cp.Variable(shape=(self.assets,), name="weights")
        self.factor_weights = cp.Variable(self.k, name="factor_weights")

        self.constraints = {}
        self._expected_returns_model = ExpectedReturns(assets=assets)
        self._risk_model = FactorModel(assets=assets, factors=k)

        self.constraints["factors"] = (
            self.factor_weights == self.risk_model.parameter["exposure"] @ self.weights
        )

    @property
    def funding(self):
        return cp.sum(self.weights)

    @property
    def leverage(self):
        return cp.norm(self.weights, 1)

    def build(self):
        objective = cp.Minimize(
            -self.expected_returns_model.estimate(self.weights)
            + cp.pos(self.risk - 0.01)
        )
        return cp.Problem(objective, list(self.constraints.values()))

    def solve(self, **kwargs):
        problem = self.build()
        problem.solve(**kwargs)
        return pd.Series(index=self.assets, data=self.weights.value)

    @property
    def risk_model(self):
        return self._risk_model

    @property
    def expected_returns_model(self):
        return self._expected_returns_model

    @property
    def risk(self):
        return self.risk_model.estimate(self.weights, y=self.factor_weights)

    def expected_return(self, returns):
        return returns @ self.weights

    def update(self, **kwargs):
        self.risk_model.update(**kwargs)
        self.expected_returns_model.update(**kwargs)


if __name__ == "__main__":
    # create a cvxpy problem
    prices = pd.read_csv(
        "data/stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    returns = prices.pct_change().dropna(axis=0, how="all")

    # compute 10 components
    components = pca(returns=returns, n_components=10)

    solver = Solver(assets=20, k=10)

    solver.update(
        cov=components.cov.values,
        exposure=components.exposure.values,
        idiosyncratic_risk=components.idiosyncratic_returns.std().values,
        mu=np.random.rand(20) / 100.0,
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )

    solver.constraints["funding"] = solver.funding == 1.0
    solver.constraints["long_only"] = solver.weights >= 0
    solver.constraints["leverage"] = solver.leverage <= 5.0

    # Add extrac constraints as you please...
    problem = solver.build()

    problem.solve()

    ###################################################################
    print(solver.weights.value)
    print(np.sum(solver.weights.value))
    weights = pd.Series(index=returns.columns, data=solver.weights.value)
    print(weights)
    print(solver.risk_model.estimate(weights.values).value)
    assert problem.is_dpp()

    # todo: parameter for target volatility
