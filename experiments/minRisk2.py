# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from cvx.linalg import cholesky
from cvx.markowitz.model import ConstraintName, VariableName
from cvx.markowitz.portfolios.min_var import MinVar
from cvx.markowitz.portfolios.utils import approx

C = ConstraintName
V = VariableName

if __name__ == "__main__":
    returns = (
        pd.read_csv("data/stock_prices.csv", index_col=0, header=0, parse_dates=True)
        .pct_change()
        .dropna(axis=0, how="all")
    )

    logger.info(f"Returns: \n{returns}")
    lower_bound_assets = pd.Series(data=0.0, index=returns.columns)
    upper_bound_assets = pd.Series(data=1.0, index=returns.columns)
    holding_costs = pd.Series(data=0.0005, index=returns.columns)

    builder = MinVar(assets=20)

    builder.parameter["max_concentration"] = cp.Parameter(
        1, name="max_concentration", nonneg=True
    )

    # You can add constraints before you build the problem
    builder.constraints[C.CONCENTRATION] = (
        cp.sum_largest(builder.variables[V.WEIGHTS], 2)
        <= builder.parameter["max_concentration"]
    )

    # here we add a constraints
    # w[19] + w[17] <= 0.0001
    # w[19] + w[17] >= -0.0001

    builder.parameter["random"] = cp.Parameter(1, name="random", nonneg=True)

    row = np.zeros(20)
    row[19] = 1
    row[17] = 1

    for name, constraint in approx(
        "xxx", row @ builder.weights, 0.0, builder.parameter["random"]
    ):
        # print(constraint)
        builder.constraints[name] = constraint

    # problem = builder.build()
    # print(problem.parameter)
    # assert False

    # print(builder.constraints)
    # print(builder.parameter)
    problem = builder.build()
    print(problem.parameter)
    # assert False

    assert problem.is_dpp(), "Problem is not DPP"

    logger.info(f"Problem is DPP: {problem.is_dpp()}")
    logger.info(problem)

    ####################################################################################################################
    problem.update(
        chol=cholesky(returns.cov().values),
        lower_assets=lower_bound_assets[returns.columns].values,
        upper_assets=upper_bound_assets[returns.columns].values,
        vola_uncertainty=np.zeros(20),
        # max_concentration=np.array([0.4]),
        # weights=np.zeros(20),
        # holding_costs=holding_costs[returns.columns].values,
    )
    problem.parameter["random"].value = np.array([0.1])
    problem.parameter["max_concentration"].value = np.array([0.4])

    logger.info("Start solving problems...")
    x = problem.solve()
    logger.info(f"Minimum standard deviation: {x}")

    # print(problem.variables[V.WEIGHTS].value)
    # print(problem.problem)
    # assert False

    # logger.info(f"weights assets:\n{minvar.solution(names=returns.columns)}")

    ####################################################################################################################
    returns = returns.iloc[:, :10]

    # second solve, should be a lot faster as the problem is DPP
    problem.update(
        chol=cholesky(returns.cov().values),
        lower_assets=lower_bound_assets[returns.columns].values,
        upper_assets=upper_bound_assets[returns.columns].values,
        vola_uncertainty=np.zeros(10),
        # weights=np.zeros(10),
        # holding_costs=holding_costs[returns.columns].values,
    )
    problem.parameter["random"].value = np.array([0.1])

    x = problem.solve()
    logger.info(f"Minimum standard deviation: {x}")
    # logger.info(f"Solution:\n{minvar.solution(returns.columns)}")
    logger.info(f"{problem}")
    logger.info(f"Concentration: {cp.sum_largest(problem.weights, 2).value}")
