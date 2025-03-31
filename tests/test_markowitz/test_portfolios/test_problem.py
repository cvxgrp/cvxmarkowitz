import cvxpy as cp
import numpy as np

from cvx.linalg.cholesky import cholesky
from cvx.linalg.random import rand_cov
from cvx.markowitz.builder import deserialize
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.portfolios.min_var import MinVar


def test_problem_data(tmp_path):
    problem = MinVar(assets=10).build()
    data, solving_chain, inverse_data = problem.get_problem_data(
        solver=cp.ECOS, verbose=True
    )
    assert data
    assert solving_chain
    assert inverse_data


def test_serialize(tmp_path):
    problem = MinVar(assets=10).build()

    problem.serialize(tmp_path / "problem.pkl")
    problem_recovered = deserialize(tmp_path / "problem.pkl")

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
