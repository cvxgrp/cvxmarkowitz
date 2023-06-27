# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz.builder import Builder
from cvx.markowitz.risk import SampleCovariance


@dataclass(frozen=True)
class DummyBuilder(Builder):
    @property
    def objective(self):
        return cp.Maximize(0.0)


def test_dummy():
    builder = DummyBuilder(assets=1)
    builder.model["risk"] = SampleCovariance(assets=1)
    builder.variables["weights"] = cp.Variable(1)

    builder.update(cov=np.eye(1))
    problem = builder.build()
    problem.solve()

    builder.variables["weights"].value = np.array([2.0])
    d = builder.solution(names=["a"])
    assert d["a"] == 2.0

    print(dict(builder.data))
    assert np.allclose(dict(builder.data)[("risk", "chol")].value, np.eye(1))
