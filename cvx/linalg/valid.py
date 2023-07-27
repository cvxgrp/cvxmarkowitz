# -*- coding: utf-8 -*-
"""Extract valid submatrix of a matrix"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def valid(
    matrix: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Construct the valid subset of matrix (correlation) matrix
    :param matrix: n x n matrix

    :return: Tuple of matrix boolean vector indicating if row/column
    is valid and the valid subset of the matrix
    """
    # make sure matrix  is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be quadratic")

    _valid = np.isfinite(np.diag(matrix))
    return _valid, matrix[:, _valid][_valid]
