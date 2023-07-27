# -*- coding: utf-8 -*-
import numpy as np
import numpy.typing as npt


def fill_vector(x: npt.NDArray[np.float64], num: int) -> npt.NDArray[np.float64]:
    """
    Fill a vector of length num with x
    """
    z = np.zeros(num)
    z[: len(x)] = x
    return z


def fill_matrix(
    x: npt.NDArray[np.float64], rows: int, cols: int
) -> npt.NDArray[np.float64]:
    """
    Fill a matrix of size (rows, cols) with x
    """
    # I had no luck with ndarray.resize()
    z = np.zeros((rows, cols))
    (n, m) = np.shape(x)
    z[:n, :m] = x
    return z
