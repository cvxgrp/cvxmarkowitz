# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from cvx.linalg import pca
from cvx.risk.factor import FactorModel

if __name__ == "__main__":
    # create a cvxpy problem
    prices = pd.read_csv(
        "data/stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    returns = prices.pct_change().dropna(axis=0, how="all")

    # compute 10 components
    components = pca(returns=returns, n_components=10)

    model = FactorModel(assets=20, k=10)

    model.update(
        cov=components.cov.values,
        exposure=components.exposure.values,
        idiosyncratic_risk=components.idiosyncratic.std().values,
    )

    weights = cp.Variable(20)
    factor_weights = cp.Variable(10)

    mu = np.random.rand(20) / 100.0

    objective = cp.Minimize(
        -mu @ weights + cp.pos(model.estimate(weights, y=factor_weights) - 0.01)
    )
    constraints = [cp.sum(weights) == 1.0, weights >= 0] + model.constraints(
        weights, y=factor_weights
    )

    #    factor_weights == model.exposure @ weights,
    # ]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    print(weights.value)
    print(np.sum(weights.value))
    weights = pd.Series(index=returns.columns, data=weights.value)
    print(weights)
    print(model.estimate(weights.values).value)
    assert problem.is_dpp()

    # fill in data
