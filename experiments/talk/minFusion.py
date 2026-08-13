"""Time the same long-only min-variance problem expressed directly in MOSEK Fusion.

Run as a script. Serves as the native-API baseline for the cvxpy timings in
`minVariance.py`: the user has to state the conic form themselves, and for this
example it comes out slower than Clarabel and ECOS.
"""

from __future__ import annotations

import time
from typing import Any

import mosek.fusion as m
import numpy as np


def min_var(cov: np.ndarray) -> tuple[np.ndarray, Any]:
    """Solve the long-only minimum-variance problem via MOSEK Fusion.

    Args:
        cov: Covariance matrix.

    Returns:
        The optimal weights and the interior-point solution status.
    """
    n = cov.shape[0]
    chol_upper = np.transpose(np.linalg.cholesky(cov))
    with m.Model() as model:
        x = model.variable(n)
        t = model.variable()
        u = chol_upper
        # doesn't help:
        # u = model.parameter('U', n, n)
        # u.setValue(chol_upper)

        model.objective(m.ObjectiveSense.Minimize, t)

        res = m.Expr.mul(u, x)
        model.constraint(m.Expr.vstack(t, res), m.Domain.inQCone())

        model.constraint("budget", m.Expr.sum(x), m.Domain.equalsTo(1.0))
        model.constraint("longonly", x, m.Domain.greaterThan(0.0))
        model.solve()
        return x.level(), model.getProblemStatus(m.SolutionType.Interior)


if __name__ == "__main__":
    n = 20
    rng = np.random.default_rng()
    cov = rng.random((n, n)) @ rng.random((n, n)).T

    # check all eigenvalues are positive
    if not np.all(np.linalg.eigh(cov).eigenvalues > 0):
        raise ValueError("covariance matrix is not positive definite")  # noqa: TRY003

    min_var(cov)

    t1 = time.time()
    for _ in range(2000):
        min_var(cov=cov)
    print(f"Solve 2000 systems, redefining the problem {time.time() - t1:.6f} seconds")
