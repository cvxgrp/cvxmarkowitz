# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.models.expected_returns import ExpectedReturns


def test_expected_returns():
    assets = 3
    model = ExpectedReturns(assets=assets)
    model.update(mu=np.array([0.1, 0.2]))

    # expected returns not explicitly set are zero
    assert model.data["mu"].value == pytest.approx(np.array([0.1, 0.2, 0.0]))

    weights = cp.Variable(assets)
    weights.value = np.array([1.0, 1.0, 2.0])
    variables = {"weights": weights}

    assert model.estimate(variables).value == pytest.approx(0.3)

    # give a new shorter vector of expected returns
    model.update(mu=np.array([0.1]))
    assert model.estimate(variables).value == pytest.approx(0.1)
