from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.models.trading_costs import TradingCosts
from cvx.markowitz.names import DataNames as D


def test_trading_costs():
    assets = 3
    model = TradingCosts(assets=assets)

    model.update(**{D.WEIGHTS: np.array([0.1, 0.2])})

    # weights not explicitly set are zero
    assert model.data[D.WEIGHTS].value == pytest.approx(np.array([0.1, 0.2, 0.0]))

    # here it's important that the weights
    weights = cp.Variable(assets)
    weights.value = np.array([0.4, 0.7, 0.0])

    variables = {D.WEIGHTS: weights}
    assert model.estimate(variables).value == pytest.approx(0.8)
