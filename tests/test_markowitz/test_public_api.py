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
        "CvxBuildError",
        "CvxDataError",
        "CvxError",
        "CvxSolverError",
        "MaxSharpe",
        "MinVar",
        "Problem",
        "SoftRisk",
    }
    assert expected <= set(cvxmarkowitz.__all__)


def test_error_subclasses_derive_from_the_base():
    """Catching CvxError still catches every specific error the package raises."""
    for subclass in (
        cvxmarkowitz.CvxBuildError,
        cvxmarkowitz.CvxDataError,
        cvxmarkowitz.CvxSolverError,
    ):
        assert issubclass(subclass, cvxmarkowitz.CvxError)


def test_every_exported_exception_derives_from_cvxerror():
    """The README's promise, enforced: no exported error sits outside the tree.

    Guards against a new exception being exported without being folded into the
    hierarchy, which is what would quietly falsify "except CvxError catches all
    of it" in the README.
    """
    exported = [getattr(cvxmarkowitz, name) for name in cvxmarkowitz.__all__]
    errors = [obj for obj in exported if isinstance(obj, type) and issubclass(obj, BaseException)]

    assert errors, "expected at least one exported exception"
    for error in errors:
        assert issubclass(error, cvxmarkowitz.CvxError), f"{error.__name__} is not a CvxError"


def test_build_returns_the_exported_problem_type():
    """Builder.build() returns the same Problem class the package exports."""
    problem = cvxmarkowitz.MinVar(assets=3).build()
    assert isinstance(problem, cvxmarkowitz.Problem)
