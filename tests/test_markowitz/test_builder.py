# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.builder import Builder
from cvx.markowitz.builder import CvxError


@dataclass(frozen=True)
class DummyBuilder(Builder):
    @property
    def objective(self):
        return cp.Maximize(0.0 + 0.0 * self.model["risk"].estimate(self.variables))


def test_dummy():
    builder = DummyBuilder(assets=1)

    assert "risk" in builder.model
    assert "bound_assets" in builder.model
    assert "chol" in builder.model["risk"].data
    assert "vola_uncertainty" in builder.model["risk"].data

    problem = builder.build()
    print(problem.problem.parameters())
    # todo: risk model needs to be involved in DummyBuilder

    problem.update(
        chol=np.eye(1), lower_assets=np.array([0.0]), upper_assets=np.array([1.0]), vola_uncertainty=np.zeros(1)
    )

    # problem = builder.build()
    problem.solve()

    problem.problem.var_dict["weights"].value = np.array([2.0])

    d = problem.solution()  # (names=["a"])
    assert d == np.array([2.0])

    print(dict(builder.data))
    assert np.allclose(dict(builder.data)[("risk", "chol")].value, np.eye(1))


def test_missing_data():
    builder = DummyBuilder(assets=1)
    problem = builder.build()
    with pytest.raises(CvxError):
        problem.update(cov=np.eye(1))
