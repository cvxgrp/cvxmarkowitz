# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from cvx.linalg import cholesky
from cvx.markowitz.portfolios.max_sharpe import MaxSharpe


def test_max_sharpe():
    # define the problem
    builder = MaxSharpe(assets=4)
    builder.parameter["sigma_max"].value = 2.0

    assert "bound_assets" in builder.model
    assert "risk" in builder.model
    assert "return" in builder.model

    problem = builder.build()

    problem.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
        mu=np.ones(2),
    )

    problem.solve()

    np.testing.assert_almost_equal(
        problem.variables["weights"].value,
        np.array([5.20124e-01, 4.79876e-01, 0.0, 0.0]),
        decimal=5,
    )

    problem.parameter["sigma_max"].value = 3.0
    problem.solve()

    np.testing.assert_almost_equal(
        problem.variables["weights"].value,
        np.array([5.10084e-01, 4.89916e-01, 0.0, 0.0]),
        decimal=5,
    )