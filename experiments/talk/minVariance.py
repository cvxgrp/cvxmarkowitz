"""Compare rebuilding a min-variance problem per solve against reusing a parametrized one.

Run as a script. For each solver it times `k` solves built from scratch, then `k`
solves that reuse a single parametrized problem whose Cholesky factor is injected
as a `cp.Parameter`.
"""

from __future__ import annotations

import time
from typing import Any

import cvxpy as cp
import numpy as np
from loguru import logger


def min_var(cov: np.ndarray, solver: str = cp.MOSEK, verbose: bool = False) -> tuple[np.ndarray, Any]:
    """Solve the long-only minimum-variance problem, rebuilding it from scratch.

    Args:
        cov: Covariance matrix.
        solver: cvxpy solver to use.
        verbose: Whether the solver should print progress.

    Returns:
        The optimal weights and the solver statistics.
    """
    n = cov.shape[0]
    chol_upper = np.transpose(np.linalg.cholesky(cov))
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm2(chol_upper @ x))
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver, verbose=verbose)
    return x.value, prob.solver_stats


def min_var_reuse(
    cov: np.ndarray,
    prob: cp.Problem,
    x: cp.Variable,
    chol_param: cp.Parameter,
    solver: str = cp.MOSEK,
    verbose: bool = False,
) -> tuple[np.ndarray, Any]:
    """Solve the same problem by injecting a new Cholesky factor into an existing one.

    Args:
        cov: Covariance matrix.
        prob: The already-constructed parametrized problem.
        x: The weight variable belonging to `prob`.
        chol_param: The parameter holding the upper Cholesky factor.
        solver: cvxpy solver to use.
        verbose: Whether the solver should print progress.

    Returns:
        The optimal weights and the solver statistics.
    """
    chol_param.value = np.transpose(np.linalg.cholesky(cov))
    prob.solve(solver=solver, verbose=verbose, warm_start=False)
    return x.value, prob.solver_stats


if __name__ == "__main__":
    n = 20
    k = 200
    rng = np.random.default_rng()
    cov = rng.random((n, n)) @ rng.random((n, n)).T

    # check all eigenvalues are positive
    if not np.all(np.linalg.eigh(cov).eigenvalues > 0):
        raise ValueError("covariance matrix is not positive definite")  # noqa: TRY003

    for solver in [cp.CLARABEL, cp.MOSEK, cp.ECOS, cp.SCS]:
        logger.info("**********************************************************")
        logger.info(solver)

        t1 = time.time()
        for _ in range(k):
            min_var(cov=cov, solver=solver, verbose=False)
        logger.info(f"Solve {k} systems, Redefine problem, {time.time() - t1:.2f} seconds")

        # construct the problem once with parameters
        weights = cp.Variable(n)
        # would be good if the parameter could be an upper triangular matrix
        # rather than just a matrix
        chol_param = cp.Parameter((n, n))

        # construct the problem
        objective = cp.Minimize(cp.norm2(chol_param @ weights))
        constraints = [cp.sum(weights) == 1, weights >= 0]
        prob = cp.Problem(objective, constraints)

        # first compilation, fills the cache
        prob.get_problem_data(solver, verbose=False)
        prob.get_problem_data(solver, verbose=False)

        t1 = time.time()
        for _ in range(k):
            min_var_reuse(cov=cov, prob=prob, x=weights, chol_param=chol_param, solver=solver, verbose=False)
        logger.info(f"Solve {k} systems, Reuse problem, {time.time() - t1:.2f} seconds")
