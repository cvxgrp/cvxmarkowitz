"""Tests for the package's top-level public API surface."""

from __future__ import annotations

import cvxmarkowitz


def test_all_is_importable():
    """Every name in __all__ resolves as a top-level package attribute."""
    for name in cvxmarkowitz.__all__:
        assert hasattr(cvxmarkowitz, name), f"{name} in __all__ but not exported"


def test_all_names_are_public():
    """No underscore-prefixed name is advertised as public API."""
    assert not [name for name in cvxmarkowitz.__all__ if name.startswith("_")]


def test_documented_entry_points_are_exported():
    """The builders, the built problem and the error type are reachable directly."""
    expected = {
        "Builder",
        "CvxError",
        "MaxSharpe",
        "MinVar",
        "Problem",
        "SoftRisk",
        "deserialize",
    }
    assert expected <= set(cvxmarkowitz.__all__)


def test_build_returns_the_exported_problem_type():
    """Builder.build() returns the same Problem class the package exports."""
    problem = cvxmarkowitz.MinVar(assets=3).build()
    assert isinstance(problem, cvxmarkowitz.Problem)
