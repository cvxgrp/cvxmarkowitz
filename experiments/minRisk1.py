# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from cvx.linalg import pca as principal_components
from cvx.markowitz.risk import FactorModel


class MinVar:
    def __init__(self, assets: int, factors: int = None):
        self.assets = assets
        self.factors = factors or self.assets
        self.model = FactorModel(assets=assets, k=self.factors)
        self.weights_assets = cp.Variable(self.assets)
        self.weights_factor = cp.Variable(self.factors)
        self.constraints = {
            **{
                "long-only": self.weights_assets >= 0,
                "funding": cp.sum(self.weights_assets) == 1.0,
            },
            **self.model.constraints(self.weights_assets, y=self.weights_factor),
        }

        self.objective = cp.Minimize(
            self.model.estimate(self.weights_assets, y=self.weights_factor)
        )

    @property
    def problem(self):
        return cp.Problem(self.objective, list(self.constraints.values()))


if __name__ == "__main__":
    returns = (
        pd.read_csv("data/stock_prices.csv", index_col=0, header=0, parse_dates=True)
        .pct_change()
        .dropna(axis=0, how="all")
    )

    logger.info(f"Returns: \n{returns}")

    # compute 10 components
    pca = principal_components(returns=returns, n_components=10)
    # pca is a NamedTuple exposing the following fields:
    # ["explained_variance", "factors", "exposure", "cov", "systematic", "idiosyncratic"],
    #   - explained_variance: pd.Series
    #   - factors: pd.DataFrame
    #   - exposure: pd.DataFrame
    #   - cov: pd.DataFrame
    #   - systematic: pd.DataFrame
    #   - idiosyncratic: pd.DataFrame

    # You can define the problem for up to 25 assets and 15 factors
    minvar = MinVar(assets=25, factors=15)

    logger.info(f"Assets: {minvar.assets}")
    logger.info(f"Factors: {minvar.factors}")

    problem = minvar.problem

    assert problem.is_dpp()
    logger.info(f"Problem is DPP: {problem.is_dpp()}")

    minvar.model.update(
        cov=pca.cov.values,
        exposure=pca.exposure.values,
        idiosyncratic_risk=pca.idiosyncratic.std().values,
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )

    logger.info(f"Factor covariance: {pca.cov}")

    logger.info("Start solving problems...")
    x = problem.solve()
    logger.info(f"Minimum variance: {x}")

    # second solve, should be a lot faster as the problem is DPP
    minvar.model.update(
        cov=pca.cov.values,
        exposure=pca.exposure.values,
        idiosyncratic_risk=pca.idiosyncratic.std().values,
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )
    x = problem.solve()
    logger.info(f"Minimum variance: {x}")

    logger.info(f"weights assets:\n{minvar.weights_assets.value}")
    logger.info(f"weights factor:\n{minvar.weights_factor.value}")
    logger.info(f"{minvar.problem}")

    for name, constraint in minvar.constraints.items():
        logger.info(f"{name}: {constraint.value}")

    # todo: understand DPP
    # todo: make DPP working for very large number of parameters
    # todo: include transaction costs
    # todo: include holding costs
