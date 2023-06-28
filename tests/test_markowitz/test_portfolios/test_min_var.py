# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from cvx.linalg import cholesky
from cvx.markowitz.portfolios.min_var import MinVar


def test_min_var():
    # define the problem
    builder = MinVar(assets=4)

    assert "bound_assets" in builder.model
    assert "risk" in builder.model

    problem = builder.build()

    builder.update(
        chol=cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )

    problem.solve()

    np.testing.assert_almost_equal(
        builder.variables["weights"].value, np.array([0.75, 0.25, 0.0, 0.0]), decimal=5
    )
