"""Tests for the built Problem container."""

import dataclasses

import cvxpy as cp
import pytest

from cvxmarkowitz import MinVar


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
