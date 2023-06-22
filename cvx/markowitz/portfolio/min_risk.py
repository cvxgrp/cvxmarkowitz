# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp


def minrisk_problem(riskmodel, weights, **kwargs):
    constraints = {
        "fully invested": cp.sum(weights) == 1.0,
        "long only": weights >= 0,
    } | riskmodel.constraints(weights, **kwargs)
    # print(constraints)
    problem = cp.Problem(
        cp.Minimize(riskmodel.estimate(weights, **kwargs)), list(constraints.values())
    )

    return problem
