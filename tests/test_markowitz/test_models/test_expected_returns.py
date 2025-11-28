"""Unit tests for the ExpectedReturns model.

Covers baseline and robust formulations and input validation.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.builder import CvxError
from cvx.markowitz.models.expected_returns import ExpectedReturns
from cvx.markowitz.names import DataNames as D


def test_expected_returns():
    """Estimate plain expected return with zero uncertainty and padding behavior."""
    assets = 3
    model = ExpectedReturns(assets=assets)
    model.update(**{D.MU: np.array([0.1, 0.2]), "mu_uncertainty": np.array([0.0, 0.0])})

    # expected returns not explicitly set are zero
    assert model.data[D.MU].value == pytest.approx(np.array([0.1, 0.2, 0.0]))
    assert model.parameter["mu_uncertainty"].value == pytest.approx(np.array([0.0, 0.0, 0.0]))

    weights = cp.Variable(assets)
    weights.value = np.array([1.0, 1.0, 2.0])
    variables = {D.WEIGHTS: weights}

    assert model.estimate(variables).value == pytest.approx(0.3)

    # give a new shorter vector of expected returns
    model.update(mu=np.array([0.1]), mu_uncertainty=np.array([0.0]))
    assert model.estimate(variables).value == pytest.approx(0.1)


def test_expected_returns_robust():
    """Include uncertainty penalty and verify the robust expected return value."""
    assets = 3
    model = ExpectedReturns(assets=assets)
    model.update(mu=np.array([0.1, 0.2]), mu_uncertainty=np.array([0.01, 0.03]))

    # expected returns not explicitly set are zero
    assert model.data["mu"].value == pytest.approx(np.array([0.1, 0.2, 0.0]))
    assert model.parameter["mu_uncertainty"].value == pytest.approx(np.array([0.01, 0.03, 0.0]))

    weights = cp.Variable(assets)
    weights.value = np.array([1.0, 1.0, 2.0])
    variables = {D.WEIGHTS: weights}

    assert model.estimate(variables).value == pytest.approx(0.26)

    # give a new shorter vector of expected returns
    model.update(mu=np.array([0.1]), mu_uncertainty=np.array([0.5]))
    assert model.estimate(variables).value == pytest.approx(-0.4)


def test_mismatch():
    """Mismatched mu and mu_uncertainty lengths should raise CvxError."""
    assets = 3
    model = ExpectedReturns(assets=assets)
    with pytest.raises(CvxError):
        model.update(mu=np.array([0.1, 0.2]), mu_uncertainty=np.array([0.03]))
