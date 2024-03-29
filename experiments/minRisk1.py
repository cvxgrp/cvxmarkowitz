from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from cvx.linalg import PCA, cholesky
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.portfolios.min_var import MinVar

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
    pca = PCA(returns=returns.values, n_components=10)
    # pca is a NamedTuple exposing the following fields:
    # ["explained_variance", "factors", "exposure", "cov",
    # "systematic_returns", "idiosyncratic_returns"],
    #   - explained_variance: pd.Series
    #   - factors: pd.DataFrame
    #   - exposure: pd.DataFrame
    #   - cov: pd.DataFrame
    #   - systematic_returns: pd.DataFrame
    #   - idiosyncratic_returns: pd.DataFrame

    lower_bound_factors = pd.Series(data=-1.0, index=range(10))
    upper_bound_factors = pd.Series(data=+1.0, index=range(10))  # len(pca.factors)))

    # You can define the problem for up to 25 assets and 15 factors
    # model = FactorModel(assets=25, k=15)
    # model = SampleCovariance(assets=25)
    builder = MinVar(assets=20, factors=10)

    holding_costs = pd.Series(data=0.0005, index=returns.columns)

    problem = builder.build()
    assert problem.is_dpp(), "Problem is not DPP"

    logger.info(f"Problem is DPP: {problem.is_dpp()}")

    ###########################################################################
    # distinguish between data and parameters
    # clean up at the end, e.g. integer lots
    problem.update(
        **{
            D.CHOLESKY: cholesky(pca.cov),
            D.EXPOSURE: pca.exposure,
            D.IDIOSYNCRATIC_VOLA: pca.idiosyncratic_vola,
            D.LOWER_BOUND_ASSETS: lower_bound_assets[returns.columns].values,
            D.UPPER_BOUND_ASSETS: upper_bound_assets[returns.columns].values,
            D.LOWER_BOUND_FACTORS: lower_bound_factors.values,
            D.UPPER_BOUND_FACTORS: upper_bound_factors.values,
            D.WEIGHTS: np.zeros(20),
            D.HOLDING_COSTS: holding_costs[returns.columns].values,
            D.SYSTEMATIC_VOLA_UNCERTAINTY: np.zeros(10),
            D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.zeros(20),
        }
    )

    # minvar.parameter["kappa"].value = kappa[returns.columns].values

    logger.info(f"Factor covariance: {pca.cov}")

    logger.info("Start solving problems...")
    x = problem.solve()
    logger.info(f"Minimum standard deviation: {x}")
    # logger.info(f"weights assets:\n{minvar.variables['weights'].value}")

    ###########################################################################
    # second solve, should be a lot faster as the problem is DPP
    returns = returns.iloc[:, :10]
    pca = PCA(returns=returns.values, n_components=5)

    problem.update(
        **{
            D.CHOLESKY: cholesky(pca.cov),
            D.EXPOSURE: pca.exposure,
            D.IDIOSYNCRATIC_VOLA: pca.idiosyncratic_vola,
            D.LOWER_BOUND_ASSETS: lower_bound_assets[returns.columns].values,
            D.UPPER_BOUND_ASSETS: upper_bound_assets[returns.columns].values,
            D.LOWER_BOUND_FACTORS: lower_bound_factors[range(5)].values,
            D.UPPER_BOUND_FACTORS: upper_bound_factors[range(5)].values,
            D.WEIGHTS: np.zeros(10),
            D.HOLDING_COSTS: holding_costs[returns.columns].values,
            D.SYSTEMATIC_VOLA_UNCERTAINTY: np.zeros(5),
            D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.zeros(10),
        }
    )

    x = problem.solve(verbose=True)
    logger.info(f"Minimum standard deviation: {x}")

    logger.info(f"{problem}")
    logger.info(f"Concentration: {cp.sum_largest(problem.weights, 2).value}")
