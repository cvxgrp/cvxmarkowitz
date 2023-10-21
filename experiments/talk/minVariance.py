import time

import cvxpy as cp
import numpy as np
from loguru import logger


def min_var(cov, solver=cp.MOSEK, verbose=False):
    # compute the minimum variance portfolio
    # cov is the covariance matrix
    n = cov.shape[0]
    U = np.transpose(np.linalg.cholesky(cov))
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm2(U @ x))
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver, verbose=verbose)
    return x.value, prob.solver_stats


def f3(cov, prob, solver=cp.MOSEK, verbose=False):
    U_param.value = np.transpose(np.linalg.cholesky(cov))
    prob.solve(solver=solver, verbose=verbose, warm_start=False)

    # _, solving_chain, _ = prob.get_problem_data(solver, verbose=verbose)
    # data, inverse_data = solving_chain.apply(prob, verbose=verbose)

    # solution = solving_chain.solve_via_data(prob, data, verbose=verbose)

    # prob.unpack_results(solution, solving_chain, inverse_data)
    return x.value, prob.solver_stats


if __name__ == "__main__":
    n = 20
    k = 200
    C = np.random.rand(n, n)
    A = C @ C.T

    # check all eigenvalue are positive
    result = np.linalg.eigh(A)
    assert np.all(result.eigenvalues > 0)

    for solver in [cp.CLARABEL, cp.MOSEK, cp.ECOS, cp.SCS]:
        logger.info("**********************************************************")
        logger.info(solver)

        t1 = time.time()
        for _ in range(k):
            min_var(cov=A, solver=solver, verbose=False)
        logger.info(
            f"Solve {k} systems, Redefine problem, {time.time() - t1:.2f} seconds"
        )

        # construct the problem once with parameters
        x = cp.Variable(n)
        # would be good if the parameter could be an upper triangular matrix
        # rather than just a matrix
        U_param = cp.Parameter((n, n))

        # construct the problem
        objective = cp.Minimize(cp.norm2(U_param @ x))
        constraints = [cp.sum(x) == 1, x >= 0]
        prob = cp.Problem(objective, constraints)

        # first compilation, fills the cache
        prob.get_problem_data(solver, verbose=False)
        prob.get_problem_data(solver, verbose=False)

        t1 = time.time()
        for _ in range(k):
            www, xxx = f3(cov=A, prob=prob, solver=solver, verbose=False)
            # print(xxx.num_iters)
        logger.info(f"Solve {k} systems, Reuse problem, {time.time() - t1:.2f} seconds")
