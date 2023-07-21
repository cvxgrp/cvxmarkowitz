# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.builder import Builder, CvxError
from cvx.markowitz.model import ConstraintName as C
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import ModelName as M


@dataclass(frozen=True)
class DummyBuilder(Builder):
    @property
    def objective(self):
        return cp.Maximize(0.0 + 0.0 * self.risk.estimate(self.variables))

    def __post_init__(self):
        super().__post_init__()
        self.constraints[C.BUDGET] = cp.sum(self.weights) == 1.0


def test_dummy():
    builder = DummyBuilder(assets=1)

    assert M.RISK in builder.model
    assert M.BOUND_ASSETS in builder.model
    assert "chol" in builder.risk.data
    assert "vola_uncertainty" in builder.risk.data

    problem = builder.build()

    problem.update(
        chol=np.eye(1),
        lower_assets=np.array([0.0]),
        upper_assets=np.array([1.0]),
        vola_uncertainty=np.zeros(1),
    ).solve(solver=cp.ECOS)

    assert np.allclose(dict(problem.data)[(M.RISK, "chol")].value, np.eye(1))


def test_missing_data():
    builder = DummyBuilder(assets=1)
    problem = builder.build()
    with pytest.raises(CvxError):
        problem.update(cov=np.eye(1))


def test_infeasible_problem():
    builder = DummyBuilder(assets=1)

    problem = builder.build()

    # check out lower bound above upper bound!
    problem.update(
        **{
            D.CHOLESKY: np.eye(1),
            D.LOWER_BOUND_ASSETS: np.array([1.0]),
            D.UPPER_BOUND_ASSETS: np.array([0.0]),
            D.VOLA_UNCERTAINTY: np.zeros(1),
        }
    )

    with pytest.raises(CvxError):
        problem.solve(solver=cp.ECOS)


def test_builder_risk():
    builder = DummyBuilder(assets=1)
    assert builder.risk == builder.model[M.RISK]
