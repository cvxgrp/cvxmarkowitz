"""Tests for the built Problem container."""

import dataclasses

import cvxpy as cp
import numpy as np
import pytest
from cvx.linalg import cholesky

from cvxmarkowitz import CvxDataError, MinVar
from cvxmarkowitz.names import DataNames as D


def _data(correlation: float) -> dict[str, np.ndarray]:
    """Return a complete data payload for a two-asset MinVar problem."""
    return {
        D.CHOLESKY: cholesky(np.array([[1.0, correlation], [correlation, 2.0]])),
        D.LOWER_BOUND_ASSETS: np.zeros(2),
        D.UPPER_BOUND_ASSETS: np.ones(2),
        D.VOLA_UNCERTAINTY: np.zeros(2),
    }


def test_problem_data():
    """get_problem_data returns the compiled data, chain and inverse data."""
    problem = MinVar(assets=10).build()
    data, solving_chain, inverse_data = problem.get_problem_data(solver=cp.CLARABEL)
    assert data
    assert solving_chain
    assert inverse_data


def test_problem_is_frozen():
    """The built problem is an immutable (frozen) dataclass."""
    problem = MinVar(assets=10).build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.problem = None


def test_update_returns_none():
    """update() mutates in place, so it returns None rather than a Problem."""
    problem = MinVar(assets=2).build()
    assert problem.update(**_data(0.5)) is None


def test_update_overwrites_in_place():
    """A second update replaces the first: there is only ever one problem.

    `frozen=True` stops attribute rebinding but not mutation of the parameter
    values, and that is deliberate -- it is what lets cvxpy reuse the cached
    canonicalization. This pins the aliasing so a future switch to copy
    semantics cannot happen silently.
    """
    problem = MinVar(assets=2).build()

    problem.update(**_data(0.0))
    first = problem.solve()

    problem.update(**_data(0.9))
    second = problem.solve()

    # the same object now answers with the second dataset
    assert first != pytest.approx(second)
    assert second == pytest.approx(0.9958, abs=1e-4)

    # a genuinely independent problem needs a fresh build
    other = MinVar(assets=2).build()
    other.update(**_data(0.0))
    assert other.solve() == pytest.approx(first)


def test_factor_weights_without_factors():
    """Asking a non-factor problem for factor weights raises CvxDataError."""
    problem = MinVar(assets=2).build()
    with pytest.raises(CvxDataError, match="without 'factors'"):
        _ = problem.factor_weights
