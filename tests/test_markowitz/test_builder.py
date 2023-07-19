# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.builder import Builder, CvxError
from cvx.markowitz.model import ConstraintName, ModelName, VariableName

C = ConstraintName
M = ModelName


@dataclass(frozen=True)
class DummyBuilder(Builder):
    @property
    def objective(self):
        return cp.Maximize(
            0.0 + 0.0 * self.model[ModelName.RISK].estimate(self.variables)
        )

    def __post_init__(self):
        super().__post_init__()
        self.constraints[C.BUDGET] = cp.sum(self.variables[VariableName.WEIGHTS]) == 1.0


def test_dummy():
    builder = DummyBuilder(assets=1)

    assert M.RISK in builder.model
    assert M.BOUND_ASSETS in builder.model
    assert "chol" in builder.model[ModelName.RISK].data
    assert "vola_uncertainty" in builder.model[ModelName.RISK].data

    problem = builder.build()
    # print(problem.problem.parameters())
    # todo: risk model needs to be involved in DummyBuilder

    problem.update(
        chol=np.eye(1),
        lower_assets=np.array([0.0]),
        upper_assets=np.array([1.0]),
        vola_uncertainty=np.zeros(1),
    ).solve(solver=cp.ECOS)

    problem.problem.var_dict["weights"].value = np.array([2.0])

    d = problem.solution()
    assert d == np.array([2.0])

    print(dict(problem.data))
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
        chol=np.eye(1),
        lower_assets=np.array([1.0]),
        upper_assets=np.array([0.0]),
        vola_uncertainty=np.zeros(1),
    )

    with pytest.raises(CvxError):
        problem.solve(solver=cp.ECOS)
