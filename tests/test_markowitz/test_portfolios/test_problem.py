"""Tests for serialization/deserialization of Markowitz problems."""

import dataclasses

import numpy as np
import pytest
from cvx.linalg import cholesky, rand_cov

from cvxmarkowitz.builder import CvxError, deserialize
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.portfolios.min_var import MinVar


def test_problem_is_frozen():
    """The built problem is an immutable (frozen) dataclass."""
    problem = MinVar(assets=10).build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.problem = None


def test_deserialize_requires_trusted(tmp_path):
    """Deserialize refuses to unpickle unless trusted=True is passed.

    Args:
        tmp_path: Pytest temporary directory for storing the pickle file.
    """
    problem = MinVar(assets=10).build()
    path = tmp_path / "problem.pkl"
    problem.serialize(path)

    with pytest.raises(CvxError, match="trusted=True"):
        deserialize(path)

    # The default is False, so a positional/implicit call is also refused.
    with pytest.raises(CvxError):
        deserialize(path, trusted=False)


def test_serialize(tmp_path):
    """Serialize a problem, deserialize it, and compare resulting weights.

    Args:
        tmp_path: Pytest temporary directory for storing the pickle file.
    """
    problem = MinVar(assets=10).build()
    problem.serialize(tmp_path / "problem.pkl")
    problem_recovered = deserialize(tmp_path / "problem.pkl", trusted=True)

    covariance = rand_cov(10)

    input_data = {
        D.CHOLESKY: cholesky(covariance),
        D.LOWER_BOUND_ASSETS: np.zeros(10),
        D.UPPER_BOUND_ASSETS: np.ones(10),
        D.VOLA_UNCERTAINTY: np.zeros(10),
    }

    problem.update(**input_data)

    problem.solve()
    sol1 = problem.weights

    problem_recovered.update(**input_data)
    problem_recovered.solve()
    sol2 = problem_recovered.weights

    np.testing.assert_array_equal(sol1, sol2)
