# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from aux.portfolio.min_var import MinVar

from cvx.linalg import cholesky
from cvx.markowitz.risk import SampleCovariance


def test_sample():
    riskmodel = SampleCovariance(assets=2)
    riskmodel.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )
    vola = riskmodel.estimate({"weights": np.array([1.0, 1.0])}).value
    np.testing.assert_almost_equal(vola, 2.0)


def test_sample_large():
    riskmodel = SampleCovariance(assets=4)
    riskmodel.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )
    vola = riskmodel.estimate({"weights": np.array([1.0, 1.0, 0.0, 0.0])}).value
    np.testing.assert_almost_equal(vola, 2.0)


def test_min_variance():
    # define the problem
    builder = MinVar(assets=4)

    assert "bound_assets" in builder.model
    assert "risk" in builder.model

    builder.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )

    problem = builder.build()
    problem.solve()

    np.testing.assert_almost_equal(
        builder.variables["weights"].value, np.array([0.75, 0.25, 0.0, 0.0]), decimal=5
    )

    # It's enough to only update the value for the cholesky decomposition
    builder.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 4.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )

    problem.solve()

    np.testing.assert_almost_equal(
        builder.variables["weights"].value,
        np.array([0.875, 0.125, 0.0, 0.0]),
        decimal=5,
    )
