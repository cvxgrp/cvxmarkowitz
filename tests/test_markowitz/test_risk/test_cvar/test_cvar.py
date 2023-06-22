# -*- coding: utf-8 -*-
# Import necessary libraries
from __future__ import annotations

import numpy as np
import pytest

from cvx.markowitz.portfolio.min_risk import minrisk_problem
from cvx.markowitz.risk import CVar


def test_estimate_risk():
    """Test the estimate() method"""
    model = CVar(alpha=0.95, n=50, assets=14)

    np.random.seed(42)

    # define the problem
    weights, _ = model.variables()
    prob = minrisk_problem(model, weights)
    assert prob.is_dpp()

    model.update(
        returns=np.random.randn(50, 10),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )
    prob.solve()
    assert prob.value == pytest.approx(0.5058720677762698)

    # it's enough to only update the R value...
    model.update(
        returns=np.random.randn(50, 10),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )
    prob.solve()
    assert prob.value == pytest.approx(0.43559171295408616)
