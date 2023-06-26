# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
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
    lower_bound_assets = pd.Series(data=0.0, index=returns.columns)
    upper_bound_assets = pd.Series(data=1.0, index=returns.columns)

    # compute 10 components
    pca = principal_components(returns=returns, n_components=10)
    # pca is a NamedTuple exposing the following fields:
    # ["explained_variance", "factors", "exposure", "cov", "systematic_returns", "idiosyncratic_returns"],
    #   - explained_variance: pd.Series
    #   - factors: pd.DataFrame
    #   - exposure: pd.DataFrame
    #   - cov: pd.DataFrame
    #   - systematic_returns: pd.DataFrame
    #   - idiosyncratic_returns: pd.DataFrame

    lower_bound_factors = pd.Series(data=-1.0, index=pca.factor_names)
    upper_bound_factors = pd.Series(data=+1.0, index=pca.factor_names)

    # You can define the problem for up to 25 assets and 15 factors
    # model = FactorModel(assets=25, k=15)
    # model = SampleCovariance(assets=25)
    minvar = MinVar(assets=20, factors=10)

    # You can add constraints before you build the problem
    # minvar.constraints["concentration"] = (
    #    cp.sum_largest(minvar.weights_assets, 2) <= 0.4
    # )
    # this constraint is not needed as the problem is long only and fully-invested
    # minvar.constraints["leverage"] = cp.abs(minvar.weights_assets) <= 3.0

    problem = minvar.build()
    assert problem.is_dpp(), "Problem is not DPP"

    logger.info(f"Problem is DPP: {problem.is_dpp()}")

    ###########################################################################
    # distinguish between data and parameters
    # clean up at the end, e.g. integer lots
    minvar.update(
        cov=pca.cov[pca.factor_names].loc[pca.factor_names].values,
        exposure=pca.exposure[pca.asset_names].loc[pca.factor_names].values,
        idiosyncratic_risk=pca.idiosyncratic_returns[pca.asset_names].std().values,
        lower_assets=lower_bound_assets[pca.asset_names].values,
        upper_assets=upper_bound_assets[pca.asset_names].values,
        lower_factors=lower_bound_factors[pca.factor_names].values,
        upper_factors=upper_bound_factors[pca.factor_names].values,
    )

    logger.info(f"Factor covariance: {pca.cov}")

    logger.info("Start solving problems...")
    x = problem.solve()
    logger.info(f"Minimum standard deviation: {x}")
    logger.info(f"weights assets:\n{minvar.weights_assets.value}")

    for name, parameter in minvar.model.data.items():
        logger.info(f"{name}: {parameter.value}")

    ###########################################################################
    # second solve, should be a lot faster as the problem is DPP
    returns = returns.iloc[:, :10]
    pca = principal_components(returns=returns, n_components=5)

    minvar.update(
        cov=pca.cov[pca.factor_names].loc[pca.factor_names].values,
        exposure=pca.exposure[pca.asset_names].loc[pca.factor_names].values,
        idiosyncratic_risk=pca.idiosyncratic_returns[pca.asset_names].std().values,
        lower_assets=lower_bound_assets[pca.asset_names].values,
        upper_assets=upper_bound_assets[pca.asset_names].values,
        lower_factors=lower_bound_factors[pca.factor_names].values,
        upper_factors=upper_bound_factors[pca.factor_names].values,
    )

    # for name, parameter in minvar.model.data.items():
    #    logger.info(f"{name}: {parameter.value}")

    # for name, parameter in minvar.model.bounds_assets.data.items():
    #    logger.info(f"{name}: {parameter.value}")

    # for name, parameter in minvar.model.bounds_factors.data.items():
    #    logger.info(f"{name}: {parameter.value}")

    x = problem.solve(verbose=True)
    logger.info(f"Minimum standard deviation: {x}")

    # logger.info(f"weights assets:\n{pd.Series(data=minvar.weights_assets.value, index=pca.asset_names)}")
    logger.info(f"Solution:\n{minvar.solution(pca.asset_names)}")
    logger.info(f"{problem}")
    logger.info(f"Concentration: {cp.sum_largest(minvar.weights_assets, 2).value}")
