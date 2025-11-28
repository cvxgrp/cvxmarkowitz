"""Experiment script to run a soft‑risk Markowitz portfolio example.

This script demonstrates how to assemble a factor‑based portfolio with a
soft risk penalty that trades off expected return versus tracking error.
It is intended for local experimentation; it does not execute any trades.

Run from the repository root, for example:

    python experiments/softRisk.py --path experiments/data/stock_prices.csv
"""

from __future__ import annotations

import cvxpy as cp
import fire
import numpy as np
import pandas as pd
from loguru import logger

from cvx.linalg import PCA, cholesky
from cvx.markowitz.models.expected_returns import ExpectedReturns
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import ModelName as M
from cvx.markowitz.names import ParameterName as P
from cvx.markowitz.portfolios.soft_risk import SoftRisk


def run(path: str = "data/stock_prices.csv") -> None:
    """Build and parameterize a soft‑risk portfolio, then prepare data.

    Args:
        path: CSV path containing price data with a DateTime index and
            one column per asset. Defaults to "data/stock_prices.csv".

    Side Effects:
        Logs intermediate information and constructs a cvxpy problem.
        This function is primarily for demonstration and does not return
        a value.
    """
    returns = pd.read_csv(path, index_col=0, header=0, parse_dates=True).pct_change().dropna(axis=0, how="all")
    n_components = 10

    assets = returns.shape[1]

    logger.info(f"Returns: \n{returns}")
    pca = PCA(returns=returns.values, n_components=n_components)
    lower_bound_factors = pd.Series(data=-1.0, index=range(n_components))
    upper_bound_factors = pd.Series(data=+1.0, index=range(n_components))  # len(pca.factors)))

    lower_bound_assets = pd.Series(data=0.0, index=returns.columns)
    upper_bound_assets = pd.Series(data=1.0, index=returns.columns)

    builder = SoftRisk(assets=assets, factors=n_components)
    builder.model[M.RETURN] = ExpectedReturns(assets=assets)

    builder.parameter[P.SIGMA_MAX] = cp.Parameter(1, name="sigma_max", nonneg=True)

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
            D.LOWER_BOUND_FACTORS: lower_bound_factors.values,
            D.UPPER_BOUND_FACTORS: upper_bound_factors.values,
            D.LOWER_BOUND_ASSETS: lower_bound_assets[returns.columns].values,
            D.UPPER_BOUND_ASSETS: upper_bound_assets[returns.columns].values,
            D.WEIGHTS: np.zeros(assets),
            D.SYSTEMATIC_VOLA_UNCERTAINTY: np.zeros(n_components),
            D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.zeros(assets),
            P.SIGMA_MAX: 0.2,
            D.MU: np.mean(returns, axis=0).values,
            D.MU_UNCERTAINTY: np.zeros(assets),
        }
    )


if __name__ == "__main__":
    fire.Fire(run)
