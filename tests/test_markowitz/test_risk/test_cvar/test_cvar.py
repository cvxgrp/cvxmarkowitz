"""Tests for the CVaR risk model used within portfolio builders.

Validates integration with the MinVar builder and checks solve values for
randomized input data.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxmarkowitz import CvxDataError
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.portfolios.min_var import MinVar
from cvxmarkowitz.risk import CVar


def test_estimate_risk(solver):
    """Smoke-test CVaR integration and objective values across updates."""
    model = CVar(alpha=0.95, rows=50, assets=14)

    np.random.seed(42)

    # Inject the risk model through the constructor. The builder only defaults
    # M.RISK when the caller did not supply one, so no SampleCovariance is built.
    builder = MinVar(assets=14, model={M.RISK: model})

    assert builder.risk is model
    assert M.BOUND_ASSETS in builder.model

    problem = builder.build()

    problem.update(
        **{
            D.RETURNS: np.random.randn(50, 10),
            D.LOWER_BOUND_ASSETS: np.zeros(10),
            D.UPPER_BOUND_ASSETS: np.ones(10),
        }
    )

    # problem = builder.build()
    problem.solve(solver=solver)
    assert problem.value == pytest.approx(0.50587206, abs=1e-5)

    problem.update(
        **{
            D.RETURNS: np.random.randn(50, 10),
            D.LOWER_BOUND_ASSETS: np.zeros(10),
            D.UPPER_BOUND_ASSETS: np.ones(10),
        }
    )

    problem.solve(solver=solver)
    assert problem.value == pytest.approx(0.4355917, abs=1e-5)


def test_injected_risk_model_replaces_the_default():
    """A risk model passed to the constructor is kept, not overwritten."""
    model = CVar(alpha=0.95, rows=50, assets=14)
    builder = MinVar(assets=14, model={M.RISK: model})

    assert builder.model[M.RISK] is model
    # the sample-covariance default would have registered a cholesky parameter
    assert D.CHOLESKY not in builder.risk.data
    assert D.RETURNS in builder.risk.data


def test_default_risk_model_is_used_when_none_is_given():
    """Without an injected model the factors-based default still applies."""
    assert D.CHOLESKY in MinVar(assets=14).risk.data
    assert D.EXPOSURE in MinVar(assets=14, factors=3).risk.data


@pytest.mark.parametrize(
    ("alpha", "rows"),
    [
        (0.95, 10),  # int(10 * 0.05) == 0
        (0.95, 0),  # the bare `rows` default
        (1.0, 100),  # no tail at all
    ],
)
def test_empty_tail_is_rejected_at_construction(alpha, rows):
    """An (alpha, rows) pair leaving no left tail must raise CvxDataError.

    Without the guard the zero tail size reaches cvxpy as a bare ValueError from
    `sum_smallest`, which is outside the CvxError tree and names neither field.
    """
    with pytest.raises(CvxDataError, match="left tail"):
        CVar(assets=4, alpha=alpha, rows=rows)


def test_empty_tail_error_names_both_fields():
    """The message must point at the two fields the caller can actually change."""
    with pytest.raises(CvxDataError) as excinfo:
        CVar(assets=4, alpha=0.95, rows=10)

    assert "alpha=0.95" in str(excinfo.value)
    assert "rows=10" in str(excinfo.value)


def test_smallest_admissible_tail_is_accepted():
    """A single scenario in the tail is enough; the guard must not over-reject."""
    model = CVar(assets=4, alpha=0.95, rows=20)  # int(20 * 0.05) == 1

    assert D.RETURNS in model.data
