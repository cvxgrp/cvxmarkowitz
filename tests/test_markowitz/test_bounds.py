# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.models.bounds import Bounds
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import VariableName as V


def test_raise_not_implemented():
    weights = cp.Variable(3)
    bounds = Bounds(assets=3, name="assets")

    with pytest.raises(NotImplementedError):
        bounds.estimate(weights)


def test_constraints():
    variables = {V.WEIGHTS: cp.Variable(3)}
    bounds = Bounds(assets=3, name="assets", acting_on=V.WEIGHTS)

    # todo: Update with DataNames
    bounds.update(
        lower_assets=np.array([0.1, 0.2]), upper_assets=np.array([0.3, 0.4, 0.5])
    )

    assert bounds.data[D.LOWER_BOUND_ASSETS].value == pytest.approx(
        np.array([0.1, 0.2, 0])
    )
    assert bounds.data[D.UPPER_BOUND_ASSETS].value == pytest.approx(
        np.array([0.3, 0.4, 0.5])
    )

    assert len(bounds.constraints(variables)) == 2


def test_wrong_action_on():
    variables = {V.WEIGHTS: cp.Variable(3)}
    bounds = Bounds(assets=3, name="assets", acting_on="wrong")

    with pytest.raises(KeyError):
        bounds.constraints(variables)
