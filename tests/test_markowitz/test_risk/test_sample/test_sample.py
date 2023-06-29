# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from cvx.linalg import cholesky
from cvx.markowitz.portfolios.min_var import MinVar
from cvx.markowitz.risk import SampleCovariance


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
        {"weights": np.array([1.0, 1.0]), "dummy": np.array([1.0, 1.0])}
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
            "weights": np.array([1.0, 1.0, 0.0, 0.0]),
            "dummy": np.array([1.0, 1.0, 0.0, 0.0]),
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
        {"weights": np.array([1.0, -1.0]), "dummy": np.array([1.0, 1.0])}
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
            "weights": np.array([1.0, -1.0, 0.0, 0.0]),
            "dummy": np.array([1.0, 1.0, 0.0, 0.0]),
        }
    ).value

    np.testing.assert_almost_equal(vola, np.sqrt(2.09))


def test_min_variance():
    # define the problem
    builder = MinVar(assets=4)

    assert "bound_assets" in builder.model
    assert "risk" in builder.model

    problem = builder.build()

    problem.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.zeros(2),
    )

    # problem = builder.build()
    problem.solve()

    np.testing.assert_almost_equal(
        problem.variables["weights"].value, np.array([0.75, 0.25, 0.0, 0.0]), decimal=5
    )

    # It's enough to only update the value for the cholesky decomposition
    problem.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 4.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        vola_uncertainty=np.zeros(2),
    )

    problem.solve()

    np.testing.assert_almost_equal(
        problem.variables["weights"].value,
        np.array([0.875, 0.125, 0.0, 0.0]),
        decimal=5,
    )
