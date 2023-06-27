# -*- coding: utf-8 -*-
"""Extract valid submatrix of a matrix"""
from __future__ import annotations

import numpy as np


def valid(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the valid subset of matrix (correlation) matrix
    :param matrix: n x n matrix

    :return: Tuple of matrix boolean vector indicating if row/column
    is valid and the valid subset of the matrix
    """
    # make sure matrix  is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    _valid = np.isfinite(np.diag(matrix))
    return _valid, matrix[:, _valid][_valid]
