# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import cvxpy as cp
import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.builder import CvxError
from cvx.markowitz.model import ModelName, VariableName
from cvx.markowitz.portfolios.min_var import MinVar
from cvx.markowitz.risk import SampleCovariance

V = VariableName


def test_sample():
    riskmodel = SampleCovariance(assets=2)
    riskmodel.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.zeros(2),
    )

    # Note: dummy should be abs(weights)
    vola = riskmodel.estimate(
        {V.WEIGHTS: np.array([1.0, 1.0]), V._ABS: np.array([1.0, 1.0])}
    ).value
    np.testing.assert_almost_equal(vola, 2.0)


def test_sample_large():
    riskmodel = SampleCovariance(assets=4)
    riskmodel.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.zeros(2),
    )
    vola = riskmodel.estimate(
        {
            V.WEIGHTS: np.array([1.0, 1.0, 0.0, 0.0]),
            V._ABS: np.array([1.0, 1.0, 0.0, 0.0]),
        }
    ).value

    np.testing.assert_almost_equal(vola, 2.0)


def test_robust_sample():
    riskmodel = SampleCovariance(assets=2)
    riskmodel.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.array([0.1, 0.2]),  # Volatility uncertainty
    )

    # Note: dummy should be abs(weights)
    vola = riskmodel.estimate(
        {V.WEIGHTS: np.array([1.0, -1.0]), V._ABS: np.array([1.0, 1.0])}
    ).value
    np.testing.assert_almost_equal(vola, np.sqrt(2.09))


def test_robust_sample_large():
    riskmodel = SampleCovariance(assets=4)
    riskmodel.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.array([0.1, 0.2]),  # Volatility uncertainty
    )
    vola = riskmodel.estimate(
        {
            V.WEIGHTS: np.array([1.0, -1.0, 0.0, 0.0]),
            V._ABS: np.array([1.0, 1.0, 0.0, 0.0]),
        }
    ).value

    np.testing.assert_almost_equal(vola, np.sqrt(2.09))


@pytest.mark.parametrize("solver", [cp.ECOS, cp.MOSEK, cp.CLARABEL])
def test_min_variance(solver):
    if os.getenv("CI", False) and solver == cp.MOSEK:
        pytest.skip("Skipping MOSEK test on CI")

    # define the problem
    builder = MinVar(assets=4)

    assert ModelName.BOUND_ASSETS in builder.model
    assert ModelName.RISK in builder.model

    problem = builder.build()

    problem.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.zeros(2),
    )

    # problem = builder.build()
    problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.weights, np.array([0.75, 0.25, 0.0, 0.0]), decimal=3
    )

    # It's enough to only update the value for the cholesky decomposition
    problem.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 4.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.zeros(2),
    )

    problem.solve(solver=solver)

    np.testing.assert_almost_equal(
        problem.weights,
        np.array([0.875, 0.125, 0.0, 0.0]),
        decimal=3,
    )


def test_mismatch():
    riskmodel = SampleCovariance(assets=4)

    with pytest.raises(CvxError):
        riskmodel.update(
            chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
            lower_assets=np.zeros(2),
            upper_assets=np.ones(2),
            vola_uncertainty=np.array([0.1]),
        )
