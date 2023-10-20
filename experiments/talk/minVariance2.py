import timeit

import cvxpy as cp
import numpy as np


def f():
    L = np.linalg.cholesky(A)

    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm2(L.T @ x))
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver)


def g():
    L.value = np.linalg.cholesky(A)
    prob.solve(solver)


if __name__ == "__main__":
    n = 300
    C = np.random.rand(n, n)
    A = C @ C.T

    result = np.linalg.eigh(A)
    assert np.all(result.eigenvalues > 0)

    for solver in [cp.MOSEK, cp.CLARABEL, cp.ECOS]:
        print(solver)
        execution_time = timeit.timeit(f, number=1)
        print(f"{execution_time:.6f} seconds")

        execution_time = timeit.timeit(f, number=10)
        print(f"{execution_time:.6f} seconds")

        x = cp.Variable(n)
        L = cp.Parameter((n, n))

        objective = cp.Minimize(cp.norm2(L.T @ x))
        constraints = [cp.sum(x) == 1, x >= 0]

        prob = cp.Problem(objective, constraints)

        execution_time = timeit.timeit(g, number=1)
        print(f"{execution_time:.6f} seconds")

        execution_time = timeit.timeit(g, number=100)
        print(f"{execution_time:.6f} seconds")
