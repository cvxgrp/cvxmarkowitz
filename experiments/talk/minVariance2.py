import cvxpy as cp
import numpy as np

if __name__ == "__main__":
    n = 200
    C = np.random.rand(n, n)
    A = C @ C.T

    # print(A)
    result = np.linalg.eigh(A)
    assert np.all(result.eigenvalues > 0)

    L = np.linalg.cholesky(A)

    x = cp.Variable(n)
