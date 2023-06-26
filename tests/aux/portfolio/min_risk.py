# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp

from cvx.markowitz.bounds import Bounds


def minrisk_problem(riskmodel, variables):
    bounds = Bounds(riskmodel.assets, name="assets", acting_on="weights")

    try:
        bounds_factors = Bounds(
            riskmodel.factors, name="factors", acting_on="factor_weights"
        )
    except AttributeError:
        bounds_factors = None

    constraints = {
        "fully invested": cp.sum(variables["weights"]) == 1.0,
        "long only": variables["weights"] >= 0,
    }

    try:
        constraints |= riskmodel.constraints(variables)
        constraints |= bounds.constraints(variables)
        constraints |= bounds_factors.constraints(variables)
    except AttributeError:
        pass

    # print(constraints)
    problem = cp.Problem(
        objective=cp.Minimize(riskmodel.estimate(variables)),
        constraints=list(constraints.values()),
    )

    return problem, bounds, bounds_factors
