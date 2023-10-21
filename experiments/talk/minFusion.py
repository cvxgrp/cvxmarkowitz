import time

import mosek.fusion as m
import numpy as np


def min_var(cov):
    # compute the minimum variance portfolio
    # cov is the covariance matrix
    n = cov.shape[0]
    U = np.transpose(np.linalg.cholesky(cov))
    with m.Model() as M:
        x = M.variable(n)
        t = M.variable()
        u = U
        # doesn't help:
        # u = M.parameter('U', n,n)
        # u.setValue(U)

        M.objective(m.ObjectiveSense.Minimize, t)

        res = m.Expr.mul(u, x)
        M.constraint(m.Expr.vstack(t, res), m.Domain.inQCone())

        M.constraint("budget", m.Expr.sum(x), m.Domain.equalsTo(1.0))
        M.constraint("longonly", x, m.Domain.greaterThan(0.0))
        M.solve()
        return x.level(), M.getProblemStatus(m.SolutionType.Interior)


if __name__ == "__main__":
    n = 20
    C = np.random.rand(n, n)
    A = C @ C.T

    # check all eigenvalue are positive
    result = np.linalg.eigh(A)
    assert np.all(result.eigenvalues > 0)

    min_var(A)

    t1 = time.time()
    for _ in range(2000):
        min_var(cov=A)
    print(f"Solve 2000 systems, redefining the problem {time.time() - t1:.6f} seconds")

    # User would need to know about cones
    # Slower than Clarabel and ECOS for this example
