"""Tests for the default risk-model selection owned by cvxmarkowitz.risk."""

from __future__ import annotations

from cvxmarkowitz import MinVar
from cvxmarkowitz.names import ModelName as M
from cvxmarkowitz.risk import CVar, FactorModel, SampleCovariance, default_risk_model


def test_default_without_factors_is_sample_covariance():
    """A non-factor problem defaults to the sample covariance model."""
    model = default_risk_model(assets=5, factors=None)
    assert isinstance(model, SampleCovariance)
    assert model.assets == 5


def test_default_with_factors_is_factor_model():
    """Passing factors selects the factor model, sized for both dimensions."""
    model = default_risk_model(assets=5, factors=2)
    assert isinstance(model, FactorModel)
    assert model.assets == 5
    assert model.factors == 2


def test_builder_defaults_through_this_function():
    """A Builder with no injected risk model gets what this function returns."""
    assert isinstance(MinVar(assets=5).risk, SampleCovariance)
    assert isinstance(MinVar(assets=5, factors=2).risk, FactorModel)


def test_an_injected_model_is_not_overwritten():
    """Injecting a risk model skips the default entirely.

    `CVar` needs `rows` and `alpha`, which a builder does not carry, so it can
    never be a default -- injection is the only route, and this pins that it
    survives `__post_init__`.
    """
    builder = MinVar(assets=5, model={M.RISK: CVar(assets=5, rows=50)})
    assert isinstance(builder.risk, CVar)
