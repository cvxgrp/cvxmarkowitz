# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.model import VariableName
from cvx.markowitz.models.holding_costs import HoldingCosts

V = VariableName


def test_holding_costs():
    assets = 3
    model = HoldingCosts(assets=assets)
    model.update(holding_costs=np.array([0.1, 0.2]))

    # weights not explicitly set are zero
    assert model.data["holding_costs"].value == pytest.approx(np.array([0.1, 0.2, 0.0]))

    # here it's important that the weights
    weights = cp.Variable(assets)
    weights.value = np.array([-0.4, 0.7, 0.0])
    variables = {V.WEIGHTS: weights}

    assert model.estimate(variables).value == pytest.approx(0.04)
