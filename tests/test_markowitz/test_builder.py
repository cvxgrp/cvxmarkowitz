"""Tests for the Builder base class via a small DummyBuilder implementation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest

from cvxmarkowitz import Builder, CvxBuildError, CvxDataError, CvxSolverError
from cvxmarkowitz.names import ConstraintName as C
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ModelName as M


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


@dataclass(frozen=True)
class NotDppBuilder(Builder):
    """A Builder whose objective is DCP-compliant but deliberately not DPP.

    The product of two parameters is not affine in the parameters, so cvxpy
    cannot cache the canonicalization -- exactly the case `build` must reject.
    """

    @property
    def objective(self):
        """Return an objective that multiplies two parameters together."""
        left = cp.Parameter(nonneg=True, name="left")
        right = cp.Parameter(nonneg=True, name="right")
        return cp.Minimize(left * right * cp.sum(self.weights))

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
    )
    problem.solve(solver=cp.CLARABEL)

    assert np.allclose(dict(problem.data)[(M.RISK, "chol")].value, np.eye(1))


def test_missing_data():
    """Updating with a wrong keyword should raise CvxDataError."""
    builder = DummyBuilder(assets=1)
    problem = builder.build()
    with pytest.raises(CvxDataError):
        problem.update(cov=np.eye(1))


def test_infeasible_problem():
    """Infeasible bounds should lead to a solver failure wrapped as CvxSolverError."""
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

    with pytest.raises(CvxSolverError):
        problem.solve(solver=cp.CLARABEL)


def test_builder_is_abstract():
    """The base Builder should not be instantiable without an objective."""
    with pytest.raises(TypeError, match="objective"):
        Builder(assets=3)


def test_builder_risk():
    """The builder.risk property should reference the risk model in model dict."""
    builder = DummyBuilder(assets=1)
    assert builder.risk == builder.model[M.RISK]


def test_non_dpp_problem_raises_cvx_build_error():
    """A non-DPP problem must be rejected with a CvxError, not an AssertionError."""
    builder = NotDppBuilder(assets=2)

    with pytest.raises(CvxBuildError, match="not DPP-compliant"):
        builder.build()


def test_the_dpp_check_survives_optimized_mode():
    """The DPP guard must be a raise, not an assert.

    `assert` is stripped under `python -O`, which would silently remove the one
    invariant the parameter-caching design depends on. Run the check in a real
    optimized interpreter rather than trusting the source to stay assert-free.
    """
    program = (
        "from tests.test_markowitz.test_builder import NotDppBuilder\n"
        "from cvxmarkowitz import CvxBuildError\n"
        "try:\n"
        "    NotDppBuilder(assets=2).build()\n"
        "except CvxBuildError:\n"
        "    print('raised')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[2],
        check=True,
    )
    assert result.stdout.strip() == "raised"
