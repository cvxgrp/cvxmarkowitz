"""Tests for the built Problem container."""

import dataclasses

import cvxpy as cp
import numpy as np
import pytest
from cvx.linalg import cholesky

from cvxmarkowitz import CvxDataError, MaxSharpe, MinVar
from cvxmarkowitz.names import DataNames as D


def _data(correlation: float) -> dict[str, np.ndarray]:
    """Return a complete data payload for a two-asset MinVar problem."""
    return {
        D.CHOLESKY: cholesky(np.array([[1.0, correlation], [correlation, 2.0]])),
        D.LOWER_BOUND_ASSETS: np.zeros(2),
        D.UPPER_BOUND_ASSETS: np.ones(2),
        D.VOLA_UNCERTAINTY: np.zeros(2),
    }


def _max_sharpe_data() -> dict[str, np.ndarray]:
    """Return a complete data payload for a two-asset MaxSharpe problem.

    MaxSharpe is the case that matters here because it carries an
    `ExpectedReturns` model, the only one with a keyword held in `parameter`
    rather than `data`.
    """
    return {
        D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
        D.LOWER_BOUND_ASSETS: np.zeros(2),
        D.UPPER_BOUND_ASSETS: np.ones(2),
        D.MU: np.array([0.25, 0.30]),
        D.MU_UNCERTAINTY: np.zeros(2),
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


def test_missing_parameter_backed_keyword_raises_cvx_data_error():
    """Omitting `mu_uncertainty` raises CvxDataError, not a bare KeyError.

    `ExpectedReturns` keeps `mu_uncertainty` in `model.parameter` rather than
    `model.data`. `Problem.update` used to build its presence check from
    `model.data` alone, so this call fell through to `kwargs["mu_uncertainty"]`
    inside the model and raised `KeyError` -- outside the `CvxError` tree the
    README promises `except CvxError` catches in full.
    """
    problem = MaxSharpe(assets=2).build()
    payload = _max_sharpe_data()
    del payload[D.MU_UNCERTAINTY]

    with pytest.raises(CvxDataError, match=D.MU_UNCERTAINTY):
        problem.update(**payload)


def test_dropping_any_required_keyword_raises_cvx_data_error():
    """Every keyword a model declares is guarded, not just the ones in `data`.

    The general form of the test above. It walks `Model.keywords` instead of
    naming keys, so a model that later starts consuming a keyword without
    declaring it is caught here rather than raising `KeyError` at a caller.
    """
    problem = MaxSharpe(assets=2).build()
    required = {key for model in problem.model.values() for key in model.keywords}

    # the parameter-backed keyword must be among them; that is the whole point
    assert D.MU_UNCERTAINTY in required
    assert required == set(_max_sharpe_data())

    for omitted in sorted(required):
        payload = _max_sharpe_data()
        del payload[omitted]
        with pytest.raises(CvxDataError, match="Missing data for"):
            problem.update(**payload)
