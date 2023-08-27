import numpy as np

from cvx.markowitz.types import Matrix


def fill_vector(x: Matrix, num: int) -> Matrix:
    """
    Fill a vector of length num with x
    """
    z = np.zeros(num)
    z[: len(x)] = x
    return z


def fill_matrix(x: Matrix, rows: int, cols: int) -> Matrix:
    """
    Fill a matrix of size (rows, cols) with x
    """
    # I had no luck with ndarray.resize()
    z = np.zeros((rows, cols))
    (n, m) = np.shape(x)
    z[:n, :m] = x
    return z
