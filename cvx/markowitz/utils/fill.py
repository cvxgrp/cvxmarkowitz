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
"""Helpers to pad vectors/matrices to target shapes."""

import numpy as np

from cvx.markowitz.types import Matrix


def fill_vector(x: Matrix, num: int) -> Matrix:
    """Create a zero-padded vector with input values at the start.

    Args:
        x: Input vector whose values are placed at the beginning.
        num: Total length of the output vector.

    Returns:
        Vector of length num with x values at the start, zeros elsewhere.
    """
    z = np.zeros(num)
    z[: len(x)] = x
    return z


def fill_matrix(x: Matrix, rows: int, cols: int) -> Matrix:
    """Create a zero-padded matrix with input values in the top-left.

    Args:
        x: Input matrix placed in the top-left corner.
        rows: Total number of rows in the output matrix.
        cols: Total number of columns in the output matrix.

    Returns:
        Matrix of shape (rows, cols) with x in top-left, zeros elsewhere.
    """
    # I had no luck with ndarray.resize()
    z = np.zeros((rows, cols))
    (n, m) = np.shape(x)
    z[:n, :m] = x
    return z
