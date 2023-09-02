"""global fixtures"""
from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture(
    params=[s for s in [cp.ECOS, cp.CLARABEL] if s in cp.installed_solvers()]
)
def solver(request):
    return request.param
