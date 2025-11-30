"""global fixtures."""

from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """Resource fixture."""
    return Path(__file__).parent / "resources"


@pytest.fixture(params=[s for s in [cp.ECOS, cp.CLARABEL] if s in cp.installed_solvers()])
def solver(request):
    """Yield installed cvxpy solvers for parametrized tests.

    Selects from a small set of preferred solvers and filters to those
    available in the current environment.
    """
    return request.param
