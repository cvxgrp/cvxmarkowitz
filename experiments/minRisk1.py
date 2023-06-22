# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from cvx.linalg import pca as principal_components
from experiments.aux.min_var import MinVar

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
    # model = FactorModel(assets=25, k=15)
    # model = SampleCovariance(assets=25)
    minvar = MinVar(assets=20, factors=10)

    # You can add constraints before you build the problem
    minvar.constraints["concentration"] = (
        cp.sum_largest(minvar.weights_assets, 2) <= 0.4
    )
    # this constraint is not needed as the problem is long only and fully-invested
    minvar.constraints["leverage"] = cp.abs(minvar.weights_assets) <= 3.0

    problem = minvar.build()
    assert problem.is_dpp()

    logger.info(f"Problem is DPP: {problem.is_dpp()}")
    logger.info(problem)

    ###########################################################################
    # distinguish between data and parameters
    # clean up at the end, e.g. integer lots
    minvar.update(
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
    logger.info(f"Minimum standard deviation: {x}")

    ###########################################################################
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
    logger.info(f"Minimum standard deviation: {x}")

    logger.info(f"weights assets:\n{minvar.weights_assets.value}")
    logger.info(f"{problem}")
    logger.info(f"Concentrations {cp.sum_largest(minvar.weights_assets, 2).value}")
