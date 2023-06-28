# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.builder import Builder
from cvx.markowitz.builder import CvxError
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

    builder.update(
        chol=np.eye(1), lower_assets=np.array([0.0]), upper_assets=np.array([1.0])
    )

    problem = builder.build()
    problem.solve()

    builder.variables["weights"].value = np.array([2.0])
    d = builder.solution(names=["a"])
    assert d["a"] == 2.0

    print(dict(builder.data))
    assert np.allclose(dict(builder.data)[("risk", "chol")].value, np.eye(1))


def test_missing_data():
    builder = DummyBuilder(assets=1)
    builder.model["risk"] = SampleCovariance(assets=1)
    builder.variables["weights"] = cp.Variable(1)
    with pytest.raises(CvxError):
        builder.update(cov=np.eye(1))
