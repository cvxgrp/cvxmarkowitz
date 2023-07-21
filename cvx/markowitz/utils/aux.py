# -*- coding: utf-8 -*-
import numpy as np


def fill_vector(x, num):
    z = np.zeros(num)
    z[: len(x)] = x
    return z


def fill_matrix(x, rows, cols):
    z = np.zeros((rows, cols))
    (n, m) = np.shape(x)
    z[:n, :m] = x
    return z


if __name__ == "__main__":
    matrix = np.ones((3, 2))
    print(matrix)
    print(fill_matrix(matrix, 4, 5))
