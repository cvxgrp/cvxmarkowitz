# -*- coding: utf-8 -*-
import numpy as np


def fill_vector(x, num):
    """
    Fill a vector of length num with x
    """
    z = np.zeros(num)
    z[: len(x)] = x
    return z


def fill_matrix(x, rows, cols):
    """
    Fill a matrix of size (rows, cols) with x
    """
    # I had no luck with ndarray.resize()
    z = np.zeros((rows, cols))
    (n, m) = np.shape(x)
    z[:n, :m] = x
    return z
