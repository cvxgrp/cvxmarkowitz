import numpy as np

from cvx.linalg.types import Matrix


def rand_cov(n: int) -> Matrix:
    a = np.random.randn(n, n)
    return np.array(np.transpose(a) @ (a))
