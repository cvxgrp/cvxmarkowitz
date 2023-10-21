import timeit

import cvxpy as cp
import numpy as np


def f1():
    U = uu
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm2(U @ x))
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver, verbose=False)


def f2():
    U = uu
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm2(U @ x))
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(objective, constraints)

    data, solving_chain, inverse_data = prob.get_problem_data(solver)
    # print(data)

    data2, inverse_data2 = solving_chain.apply(prob)
    # for key in data:
    #    assert key in data2
    #    assert np.isclose(data[key], data2[key])

    # assert np.all(data[key] == data2[key])

    # assert data2==data
    # assert inverse_data2==inverse_data
    # for key in inverse_data:
    #    assert key in inverse_data2
    #    assert np.isclose(inverse_data[key], inverse_data2[key])

    # assert np.all(inverse_data[key] == inverse_data2[key])

    solution = solving_chain.solve_via_data(prob, data2)

    prob.unpack_results(solution, solving_chain, inverse_data2)
    return prob.value


# def g():
#    U.value = uu
#    prob.solve(solver)


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
        execution_time = timeit.timeit(f1, number=1)
        print(f"{execution_time:.6f} seconds")

        execution_time = timeit.timeit(f2, number=1)
        print(f"{execution_time:.6f} seconds")

        # solve 10 problems, each time the problem is reconstructed
        # Should take 10times longer than the previous one
        # execution_time = timeit.timeit(f1, number=10)
        # print(f"{execution_time:.6f} seconds")

        # construct the problem once with parameters
        # x = cp.Variable(n)
        # would be good if the parameter could be an upper triangular matrix
        # rather than just a matrix
        # U = cp.Parameter((n, n))

        # construct the problem
        # objective = cp.Minimize(cp.norm2(U @ x))
        # constraints = [cp.sum(x) == 1, x >= 0]
        # prob = cp.Problem(objective, constraints)

        # could we know construct the problem
        # xxx = prob.get_problem_data(solver)
        # what can we do with xxx?

        # solve the problem only once, should be faster
        # than the previous one, but not for CLARABEL and ECOS
        # execution_time = timeit.timeit(g, number=1)
        # print(f"{execution_time:.6f} seconds")
        # assert False

        # solve the problem 10 times
        # execution_time = timeit.timeit(g, number=10)
        # print(f"{execution_time:.6f} seconds")
