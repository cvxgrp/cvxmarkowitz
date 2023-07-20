# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from cvx.linalg.cholesky import cholesky
from cvx.markowitz.portfolios.min_var import MinVar
from cvx.simulator.builder import builder

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/stock_prices.csv", index_col=0, header=0, parse_dates=True
    )

    # --------------------------------------------------------------------------------------------
    # construct the "Markowitz engine", here use a very simple idea
    engine = MinVar(assets=20)
    # add additional constraints you like
    problem = engine.build()

    # expected data for each update...
    for tuple in problem.expected_names:
        logger.info(tuple)

    logger.info(set(problem.parameter.keys()))
    logger.info(set(problem.model.keys()))

    # --------------------------------------------------------------------------------------------
    # construct the portfolio using a builder
    b = builder(prices=prices)

    # --------------------------------------------------------------------------------------------
    # compute data needed for the portfolio construction
    cov = dict(b.cov(halflife=10, min_periods=30))

    # --------------------------------------------------------------------------------------------
    # perform the iteration through time
    for t, _ in b:
        try:
            # update the problem
            problem.update(
                chol=cholesky(cov[t[-1]].values),
                vola_uncertainty=np.zeros(20),
                lower_assets=np.zeros(20),
                upper_assets=np.ones(20),
            )

            # solve the problem
            problem.solve()
            weights = pd.Series(index=prices.columns, data=problem.weights)
            # update the builder
            b.set_weights(t[-1], weights=weights)
        except KeyError:
            pass

    # --------------------------------------------------------------------------------------------
    # build the portfolio
    portfolio = b.build()
    portfolio.snapshot()
    portfolio.metrics()
