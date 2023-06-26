# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp

from cvx.markowitz.bounds import Bounds


def minrisk_problem(riskmodel, weights, **kwargs):
    bounds = Bounds(riskmodel.assets, name="assets")
    try:
        bounds_factors = Bounds(riskmodel.factors, name="factors")
    except AttributeError:
        bounds_factors = None

    constraints = {
        "fully invested": cp.sum(weights) == 1.0,
        "long only": weights >= 0,
    }

    try:
        constraints |= riskmodel.constraints(weights, **kwargs)
        constraints |= bounds.constraints(weights)
        constraints |= bounds_factors.constraints(riskmodel.factor_weights(weights))
    except AttributeError:
        pass

    # print(constraints)
    problem = cp.Problem(
        cp.Minimize(riskmodel.estimate(weights, **kwargs)), list(constraints.values())
    )

    return problem, bounds, bounds_factors
