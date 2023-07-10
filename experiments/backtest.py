# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
from loguru import logger

from cvx.markowitz.portfolios.min_var import MinVar
from cvx.simulator.builder import builder

# from cvx.linalg import PCA, cholesky
# from cvx.markowitz.portfolios.min_var import MinVar

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/stock_prices.csv", index_col=0, header=0, parse_dates=True
    )

    logger.info(f"Loaded prices. Shape: {prices.shape}")

    b = builder(prices=prices)
    problem = MinVar(assets=20).build()

    for t, state in b:
        print(t)

    portfolio = b.build()
