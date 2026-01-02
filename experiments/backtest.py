"""Example backtest using a simple minimum-variance engine."""

from __future__ import annotations

import fire as fire
import numpy as np
import pandas as pd
from cvx.simulator.builder import builder
from loguru import logger

from cvxmarkowitz.linalg import cholesky
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.portfolios.min_var import MinVar


def run(path: str, halflife: int = 10, min_periods: int = 30) -> None:
    """Run the backtest for the given data file using a simple minimum variance engine.

    Parameters
    ----------
    path : str
        Path to the data file.
    halflife : int, optional
        Halflife of the exponential weighting function, by default 10
    min_periods : int, optional
        Minimum number of periods to compute the covariance matrix, by default 30
    """
    prices = pd.read_csv(path, index_col=0, header=0, parse_dates=True)
    n_assets = prices.shape[1]

    # --------------------------------------------------------------------------------------------
    # construct the "Markowitz engine", here use a very simple idea
    engine = MinVar(assets=n_assets)
    # add additional constraints you like
    problem = engine.build()

    # expected data for each update...
    # for tuple in problem.expected_names:
    #     logger.info(tuple)

    logger.info(set(problem.parameter.keys()))
    logger.info(set(problem.model.keys()))

    # --------------------------------------------------------------------------------------------
    # construct the portfolio using a builder
    b = builder(prices=prices)

    # --------------------------------------------------------------------------------------------
    # compute data needed for the portfolio construction
    cov = dict(b.cov(halflife=halflife, min_periods=min_periods))

    # --------------------------------------------------------------------------------------------
    # perform the iteration through time
    for t, _ in b:
        try:
            # update the problem
            problem.update(
                **{
                    D.CHOLESKY: cholesky(cov[t[-1]].values),
                    D.VOLA_UNCERTAINTY: np.zeros(n_assets),
                    D.LOWER_BOUND_ASSETS: np.zeros(n_assets),
                    D.UPPER_BOUND_ASSETS: np.ones(n_assets),
                }
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


if __name__ == "__main__":
    fire.Fire(run)
