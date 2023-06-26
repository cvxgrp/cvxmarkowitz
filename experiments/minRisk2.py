# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import pandas as pd
from loguru import logger

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

    minvar = MinVar(assets=20)

    # You can add constraints before you build the problem
    minvar.constraints["concentration"] = (
        cp.sum_largest(minvar.weights_assets, 2) <= 0.4
    )

    problem = minvar.build()
    assert problem.is_dpp(), "Problem is not DPP"

    logger.info(f"Problem is DPP: {problem.is_dpp()}")
    logger.info(problem)

    ####################################################################################################################
    minvar.update(
        cov=returns.cov().values,
        lower_assets=lower_bound_assets[returns.columns].values,
        upper_assets=upper_bound_assets[returns.columns].values,
    )

    logger.info("Start solving problems...")
    x = problem.solve()
    logger.info(f"Minimum standard deviation: {x}")
    logger.info(f"weights assets:\n{minvar.solution(names=returns.columns)}")

    ####################################################################################################################
    returns = returns.iloc[:, :10]

    # second solve, should be a lot faster as the problem is DPP
    minvar.update(
        cov=returns.cov().values,
        lower_assets=lower_bound_assets[returns.columns].values,
        upper_assets=upper_bound_assets[returns.columns].values,
    )

    x = problem.solve()
    logger.info(f"Minimum standard deviation: {x}")
    logger.info(f"Solution:\n{minvar.solution(returns.columns)}")
    logger.info(f"{problem}")
    logger.info(f"Concentration: {cp.sum_largest(minvar.weights_assets, 2).value}")
