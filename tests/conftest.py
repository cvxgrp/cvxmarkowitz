"""Global fixtures for test suite."""

from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """Provide path to test resources directory.

    Returns:
        Path to the resources directory containing test data files.
    """
    return Path(__file__).parent / "resources"


@pytest.fixture(params=[s for s in [cp.ECOS, cp.CLARABEL] if s in cp.installed_solvers()])
def solver(request):
    """Parametrized fixture providing available CVXPY solvers.

    Args:
        request: Pytest request object with parameter.

    Returns:
        Solver name string for use with cvxpy.
    """
    return request.param
