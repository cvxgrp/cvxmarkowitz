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
    """Pad a vector with zeros to reach target length.

    Args:
        x: Input vector to pad.
        num: Target length for the output vector.

    Returns:
        Zero-padded vector of length num with x values at the start.
    """
    z = np.zeros(num)
    z[: len(x)] = x
    return z


def fill_matrix(x: Matrix, rows: int, cols: int) -> Matrix:
    """Pad a matrix with zeros to reach target dimensions.

    Args:
        x: Input matrix to pad.
        rows: Target number of rows.
        cols: Target number of columns.

    Returns:
        Zero-padded matrix of shape (rows, cols) with x in the top-left.
    """
    # I had no luck with ndarray.resize()
    z = np.zeros((rows, cols))
    (n, m) = np.shape(x)
    z[:n, :m] = x
    return z
