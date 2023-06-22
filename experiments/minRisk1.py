# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from cvx.linalg import pca as principal_components
from cvx.markowitz import Model
from cvx.markowitz.risk import FactorModel
from cvx.markowitz.risk import SampleCovariance


class MinVar:
    def __init__(self, riskmodel: Model = None):
        self.model = riskmodel
        self.weights_assets, self.weights_factor = model.variables
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
        return cp.Problem(self.objective, list(self.constraints.values()))

    def update(self, **kwargs):
        self.model.update(**kwargs)


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
    model = FactorModel(assets=25, k=15)

    # model = SampleCovariance(assets=25)
    minvar = MinVar(riskmodel=model)

    logger.info(f"Assets: {minvar.model.assets}")

    # You can add constraints before you build the problem
    minvar.constraints["concentration"] = (
        cp.sum_largest(minvar.weights_assets, 2) <= 0.4
    )
    # this constraint is not needed as the problem is long only and fully-invested
    minvar.constraints["leverage"] = cp.abs(minvar.weights_assets) <= 3.0

    problem = minvar.build()
    assert problem.is_dpp()

    logger.info(f"Problem is DPP: {problem.is_dpp()}")

    # cov = returns.cov()  # else pca.cov
    cov = pca.cov

    minvar.update(
        cov=cov.values,
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
    minvar.update(
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
    logger.info(f"{problem}")

    for name, constraint in minvar.constraints.items():
        logger.info(f"{name}: {constraint.value}")

    print(cp.sum_largest(minvar.weights_assets, 2).value)

    # todo: understand DPP
    # todo: make DPP working for very large number of parameters
    # todo: include transaction costs
    # todo: include holding costs
