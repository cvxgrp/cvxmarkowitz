# -*- coding: utf-8 -*-
# Import necessary libraries
from __future__ import annotations

import numpy as np
import pytest

from cvx.markowitz.portfolios.min_var import MinVar
from cvx.markowitz.risk import CVar


def test_estimate_risk():
    """Test the estimate() method"""
    model = CVar(alpha=0.95, rows=50, assets=14)

    np.random.seed(42)

    # define the problem
    builder = MinVar(assets=14)

    # overwrite the risk model
    builder.model["risk"] = model

    assert "bound_assets" in builder.model

    builder.update(
        returns=np.random.randn(50, 10),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )

    problem = builder.build()
    problem.solve()
    assert problem.value == pytest.approx(0.5058720677762698)

    # it's enough to only update the R value...
    builder.update(
        returns=np.random.randn(50, 10),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )
    problem.solve()
    assert problem.value == pytest.approx(0.43559171295408616)
