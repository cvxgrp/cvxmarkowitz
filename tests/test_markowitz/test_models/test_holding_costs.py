"""Unit tests for the HoldingCosts model."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvxmarkowitz.models.holding_costs import HoldingCosts
from cvxmarkowitz.names import DataNames as D


def test_holding_costs():
    """Holding costs should penalize negative weight-cost products as expected."""
    assets = 3
    model = HoldingCosts(assets=assets)
    model.update(**{D.HOLDING_COSTS: np.array([0.1, 0.2])})

    # weights not explicitly set are zero
    assert model.data[D.HOLDING_COSTS].value == pytest.approx(np.array([0.1, 0.2, 0.0]))

    # here it's important that the weights
    weights = cp.Variable(assets)
    weights.value = np.array([-0.4, 0.7, 0.0])
    variables = {D.WEIGHTS: weights}

    assert model.estimate(variables).value == pytest.approx(0.04)
