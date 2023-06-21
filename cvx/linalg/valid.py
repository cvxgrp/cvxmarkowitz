# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def valid(matrix):
    """
    Construct the valid subset of matrix (correlation) matrix matrix
    :param matrix: n x n matrix

    :return: Tuple of matrix boolean vector indicating if row/column
    is valid and the valid subset of the matrix
    """
    # make sure matrix  is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    v = np.isfinite(np.diag(matrix))
    return v, matrix[:, v][v]
