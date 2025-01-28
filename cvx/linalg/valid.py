#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Extract valid submatrix of a matrix"""

from __future__ import annotations

import numpy as np

from .types import Matrix


def valid(matrix: Matrix) -> tuple[Matrix, Matrix]:
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
