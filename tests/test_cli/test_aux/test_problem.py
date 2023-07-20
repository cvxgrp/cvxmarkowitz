# -*- coding: utf-8 -*-
import numpy as np

from cvx.cli.aux.problem import deserialize_problem, serialize_problem
from cvx.linalg.cholesky import cholesky
from cvx.linalg.random import rand_cov
from cvx.markowitz.portfolios.min_var import MinVar


def test_serialize(tmp_path):
    problem = MinVar(assets=10).build()
    serialize_problem(problem, tmp_path / "problem.pkl")
    problem_recovered = deserialize_problem(tmp_path / "problem.pkl")

    # assert "long-only" in problem_recovered.constraints
    # assert "fully-invested" in problem_recovered.constraints
    # assert problem_recovered.variables.keys() == {"weights", "_abs"}
    # assert problem_recovered.parameter == {}

    covariance = rand_cov(10)

    input_data = {}
    # print(set(problem.expected_names))
    # assert set(problem.expected_names) == {
    #     "chol",
    #     "lower_assets",
    #     "upper_assets",
    #     "vola_uncertainty",
    # }

    # assert False

    input_data["chol"] = cholesky(covariance)
    input_data["lower_assets"] = np.zeros(10)
    input_data["upper_assets"] = np.ones(10)
    input_data["vola_uncertainty"] = np.zeros(10)

    problem.update(**input_data)
    sol1 = problem.solve()
    sol1 = problem.weights

    problem_recovered.update(**input_data)
    sol2 = problem_recovered.solve()
    sol2 = problem_recovered.weights

    np.testing.assert_array_equal(sol1, sol2)
