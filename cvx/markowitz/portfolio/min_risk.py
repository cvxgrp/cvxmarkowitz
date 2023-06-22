# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp


def minrisk_problem(riskmodel, weights, **kwargs):
    problem = cp.Problem(
        cp.Minimize(riskmodel.estimate(weights, **kwargs)),
        [cp.sum(weights) == 1.0, weights >= 0]
        + riskmodel.constraints(weights, **kwargs),
    )

    return problem
