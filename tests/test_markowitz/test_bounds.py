"""Tests for Bounds model updating and constraints wiring."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvxmarkowitz.models.bounds import Bounds
from cvxmarkowitz.names import DataNames as D


def test_raise_not_implemented():
    """Estimate on Bounds should raise NotImplementedError."""
    weights = cp.Variable(3)
    bounds = Bounds(assets=3, name="assets")

    with pytest.raises(NotImplementedError):
        bounds.estimate(weights)


def test_constraints():
    """Updated bounds should pad vectors and generate two constraints."""
    variables = {D.WEIGHTS: cp.Variable(3)}
    bounds = Bounds(assets=3, name="assets", acting_on=D.WEIGHTS)

    bounds.update(
        **{
            D.LOWER_BOUND_ASSETS: np.array([0.1, 0.2]),
            D.UPPER_BOUND_ASSETS: np.array([0.3, 0.4, 0.5]),
        }
    )

    assert bounds.data[D.LOWER_BOUND_ASSETS].value == pytest.approx(np.array([0.1, 0.2, 0]))
    assert bounds.data[D.UPPER_BOUND_ASSETS].value == pytest.approx(np.array([0.3, 0.4, 0.5]))

    assert len(bounds.constraints(variables)) == 2


def test_wrong_action_on():
    """Using an unknown acting_on key should raise KeyError when building constraints."""
    variables = {D.WEIGHTS: cp.Variable(3)}
    bounds = Bounds(assets=3, name="assets", acting_on="wrong")

    with pytest.raises(KeyError):
        bounds.constraints(variables)
