"""Tests for the Builder base class via a small DummyBuilder implementation."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pytest

from cvx.markowitz.builder import Builder, CvxError
from cvx.markowitz.names import ConstraintName as C
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.names import ModelName as M


@dataclass(frozen=True)
class DummyBuilder(Builder):
    """Minimal concrete Builder used to test base‑class behavior."""

    @property
    def objective(self):
        """Return a trivial objective to exercise the builder wiring in tests."""
        return cp.Maximize(0.0 + 0.0 * self.risk.estimate(self.variables))

    def __post_init__(self):
        """Initialize base components and add a unit budget constraint for tests."""
        super().__post_init__()
        self.constraints[C.BUDGET] = cp.sum(self.weights) == 1.0


def test_dummy():
    """Smoke-test building and solving a 1-asset dummy problem."""
    builder = DummyBuilder(assets=1)

    assert M.RISK in builder.model
    assert M.BOUND_ASSETS in builder.model
    assert D.CHOLESKY in builder.risk.data
    assert D.VOLA_UNCERTAINTY in builder.risk.data

    problem = builder.build()

    problem.update(
        **{
            D.CHOLESKY: np.eye(1),
            D.LOWER_BOUND_ASSETS: np.array([0.0]),
            D.UPPER_BOUND_ASSETS: np.array([1.0]),
            D.VOLA_UNCERTAINTY: np.zeros(1),
        }
    ).solve(solver=cp.CLARABEL)

    assert np.allclose(dict(problem.data)[(M.RISK, "chol")].value, np.eye(1))


def test_missing_data():
    """Updating with a wrong keyword should raise CvxError."""
    builder = DummyBuilder(assets=1)
    problem = builder.build()
    with pytest.raises(CvxError):
        problem.update(cov=np.eye(1))


def test_infeasible_problem():
    """Infeasible bounds should lead to a solver failure wrapped as CvxError."""
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
        problem.solve(solver=cp.CLARABEL)


def test_builder_risk():
    """The builder.risk property should reference the risk model in model dict."""
    builder = DummyBuilder(assets=1)
    assert builder.risk == builder.model[M.RISK]
