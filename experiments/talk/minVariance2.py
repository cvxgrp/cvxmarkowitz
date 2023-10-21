import timeit

import cvxpy as cp
import numpy as np


def f():
    U = uu
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm2(U @ x))
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver)


def g():
    U.value = uu
    prob.solve(solver)


if __name__ == "__main__":
    n = 300
    C = np.random.rand(n, n)
    A = C @ C.T

    uu = np.transpose(np.linalg.cholesky(A))

    # check all eigenvalue are positive
    result = np.linalg.eigh(A)
    assert np.all(result.eigenvalues > 0)

    for solver in [cp.MOSEK, cp.CLARABEL, cp.ECOS]:
        print(solver)

        # solve one problem
        execution_time = timeit.timeit(f, number=1)
        print(f"{execution_time:.6f} seconds")

        # solve 10 problems, each time the problem is reconstructed
        # Should take 10times longer than the previous one
        execution_time = timeit.timeit(f, number=10)
        print(f"{execution_time:.6f} seconds")

        # construct the problem once with parameters
        x = cp.Variable(n)
        # would be good if the parameter could be an upper triangular matrix
        # rather than just a matrix
        U = cp.Parameter((n, n))

        # construct the problem
        objective = cp.Minimize(cp.norm2(U @ x))
        constraints = [cp.sum(x) == 1, x >= 0]
        prob = cp.Problem(objective, constraints)

        # could we know construct the problem
        xxx = prob.get_problem_data(solver)
        # what can we do with xxx?

        # solve the problem only once, should be faster
        # than the previous one, but not for CLARABEL and ECOS
        execution_time = timeit.timeit(g, number=1)
        print(f"{execution_time:.6f} seconds")
        # assert False

        # solve the problem 10 times
        execution_time = timeit.timeit(g, number=10)
        print(f"{execution_time:.6f} seconds")
