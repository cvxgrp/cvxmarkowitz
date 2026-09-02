"""Tests for the built Problem container."""

import dataclasses

import cvxpy as cp
import numpy as np
import pytest
from cvx.linalg import cholesky

from cvxmarkowitz import CvxDataError, MaxSharpe, MinVar
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ModelName as M


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


def _factor_data(assets: int = 3, factors: int = 2) -> dict[str, np.ndarray]:
    """Return a complete payload for a MinVar problem with a factor risk model."""
    return {
        D.EXPOSURE: np.ones((factors, assets)),
        D.CHOLESKY: np.eye(factors),
        D.IDIOSYNCRATIC_VOLA: np.full(assets, 0.1),
        D.IDIOSYNCRATIC_VOLA_UNCERTAINTY: np.zeros(assets),
        D.SYSTEMATIC_VOLA_UNCERTAINTY: np.zeros(factors),
        D.LOWER_BOUND_ASSETS: np.zeros(assets),
        D.UPPER_BOUND_ASSETS: np.ones(assets),
        D.LOWER_BOUND_FACTORS: -np.ones(factors),
        D.UPPER_BOUND_FACTORS: np.ones(factors),
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


def test_a_smaller_universe_is_padded_and_solved():
    """Data for fewer assets than the problem was built for is still legal.

    This is the whole point of the padding, so it is also what the consistency
    check below must not break: `Bounds` pads both bounds with zeros, which pins
    the unused tail to `0 <= w <= 0`, and the answer is the two-asset one.
    """
    padded = MinVar(assets=4).build()
    padded.update(**_data(0.5))

    exact = MinVar(assets=2).build()
    exact.update(**_data(0.5))

    assert padded.solve() == pytest.approx(exact.solve())
    assert padded.weights[2:] == pytest.approx(np.zeros(2), abs=1e-6)


def test_models_disagreeing_about_the_universe_are_rejected():
    """A payload sizing the risk model and the bounds differently is refused.

    Left to itself this solves happily and answers with nonsense: the risk model
    pads its Cholesky factor with zeros, the bounds are given their full length,
    and the tail becomes a set of riskless assets the solver puts everything
    into. No single model can see it -- each one's own inputs are consistent.
    """
    problem = MinVar(assets=4).build()
    payload = _data(0.5) | {D.LOWER_BOUND_ASSETS: np.zeros(4), D.UPPER_BOUND_ASSETS: np.ones(4)}

    with pytest.raises(CvxDataError, match="Inconsistent size for weights"):
        problem.update(**payload)


def test_the_error_names_both_disagreeing_models():
    """The message points at the two models, since the payload cannot say which is wrong."""
    problem = MinVar(assets=4).build()
    payload = _data(0.5) | {D.LOWER_BOUND_ASSETS: np.zeros(4), D.UPPER_BOUND_ASSETS: np.ones(4)}

    with pytest.raises(CvxDataError) as excinfo:
        problem.update(**payload)

    assert M.RISK in str(excinfo.value)
    assert M.BOUND_ASSETS in str(excinfo.value)


def test_bounds_disagreeing_with_each_other_are_rejected():
    """A model's inputs are checked against each other too, not only across models.

    Both bounds are padded with zeros, so a short lower bound against a full
    upper bound leaves the tail free between 0 and its real upper bound -- the
    same riskless-tail failure, from a single model.
    """
    problem = MinVar(assets=4).build()
    payload = _data(0.5) | {D.UPPER_BOUND_ASSETS: np.ones(4)}

    with pytest.raises(CvxDataError, match="Inconsistent size for weights"):
        problem.update(**payload)


def test_factor_count_disagreement_is_rejected():
    """Factors are checked as their own dimension, independently of the assets."""
    problem = MinVar(assets=3, factors=2).build()
    payload = _factor_data() | {
        D.LOWER_BOUND_FACTORS: -np.ones(3),
        D.UPPER_BOUND_FACTORS: np.ones(3),
    }

    with pytest.raises(CvxDataError, match=f"Inconsistent size for {D.FACTOR_WEIGHTS}"):
        problem.update(**payload)


def test_a_consistent_factor_payload_still_passes():
    """The factor problem's own dimensions agree, so a good payload is untouched.

    With an exposure of ones every budgeted portfolio has the same factor
    weights, so the systematic risk is fixed at ``sqrt(2)`` and only the
    idiosyncratic part is left to minimise -- which the equal-weight portfolio
    does.
    """
    problem = MinVar(assets=3, factors=2).build()
    problem.update(**_factor_data())
    value = problem.solve()

    assert problem.weights == pytest.approx(np.full(3, 1.0 / 3.0), abs=1e-6)
    assert value == pytest.approx(np.sqrt(2.0 + 0.1**2 / 3.0), abs=1e-6)


def test_a_rejected_payload_writes_nothing():
    """Validation runs over every model before the first value is written.

    Otherwise a payload rejected on the last model would leave the problem
    half-overwritten -- neither the old dataset nor the new one.
    """
    problem = MinVar(assets=4).build()
    problem.update(**_data(0.0))
    before = problem.solve()

    with pytest.raises(CvxDataError):
        problem.update(**(_data(0.9) | {D.UPPER_BOUND_ASSETS: np.ones(4)}))

    assert problem.solve() == pytest.approx(before)


def test_a_universe_larger_than_the_compiled_one_is_rejected():
    """Padding only ever goes up, so oversized data is a CvxDataError.

    The models agree with each other here -- they agree on a universe the
    compiled problem has no room for -- so this is the `fill` guard rather than
    the cross-model one, reached through the public entry point.
    """
    problem = MinVar(assets=2).build()
    payload = {
        D.CHOLESKY: np.eye(4),
        D.VOLA_UNCERTAINTY: np.zeros(4),
        D.LOWER_BOUND_ASSETS: np.zeros(4),
        D.UPPER_BOUND_ASSETS: np.ones(4),
    }

    with pytest.raises(CvxDataError, match="does not fit a problem built for"):
        problem.update(**payload)
