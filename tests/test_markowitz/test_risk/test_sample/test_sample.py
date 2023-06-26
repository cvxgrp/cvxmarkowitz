# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from cvx.markowitz.portfolio.min_risk import minrisk_problem
from cvx.markowitz.risk import SampleCovariance


def test_sample():
    riskmodel = SampleCovariance(assets=2)
    riskmodel.update(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )
    vola = riskmodel.estimate({"weights": np.array([1.0, 1.0])}).value
    np.testing.assert_almost_equal(vola, 2.0)


def test_sample_large():
    riskmodel = SampleCovariance(assets=4)
    riskmodel.update(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )
    vola = riskmodel.estimate({"weights": np.array([1.0, 1.0, 0.0, 0.0])}).value
    np.testing.assert_almost_equal(vola, 2.0)


def test_min_variance():
    riskmodel = SampleCovariance(assets=4)

    variables = riskmodel.variables
    problem, bounds, _ = minrisk_problem(riskmodel, variables)
    assert problem.is_dpp()

    riskmodel.update(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
    )

    bounds.update(
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )

    problem.solve()
    np.testing.assert_almost_equal(
        variables["weights"].value, np.array([0.75, 0.25, 0.0, 0.0]), decimal=5
    )

    # It's enough to only update the value for the cholesky decomposition
    riskmodel.update(
        cov=np.array([[1.0, 0.5], [0.5, 4.0]]),
    )

    problem.solve()
    np.testing.assert_almost_equal(
        variables["weights"].value, np.array([0.875, 0.125, 0.0, 0.0]), decimal=5
    )
