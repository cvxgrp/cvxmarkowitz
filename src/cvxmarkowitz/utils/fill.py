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

from cvxmarkowitz.cvxerror import CvxDataError
from cvxmarkowitz.types import Matrix


def fill_vector(x: Matrix, num: int) -> Matrix:
    """Return a vector of length ``num`` holding ``x`` in its leading entries.

    The tail is zero. This is what lets one compiled problem serve a universe
    smaller than the one it was built for: `Bounds` pads both of its bounds
    this way, which pins the unused tail to ``0 <= w <= 0``.

    Padding only ever goes one way. An ``x`` longer than ``num`` does not fit
    the compiled problem at all, so it is reported as a `CvxDataError` rather
    than truncated silently or left to escape as the `ValueError` numpy raises
    on the assignment below -- `CvxDataError` is the failure mode the README
    promises for input whose shapes do not fit.

    Raises:
        CvxDataError: If ``x`` is longer than ``num``.
    """
    if len(x) > num:
        raise CvxDataError(f"Vector of length {len(x)} does not fit a problem built for {num}")  # noqa: TRY003

    z = np.zeros(num)
    z[: len(x)] = x
    return z


def fill_matrix(x: Matrix, rows: int, cols: int) -> Matrix:
    """Return a ``rows`` x ``cols`` matrix holding ``x`` in its top-left block.

    The counterpart of `fill_vector`; see there for why the padding is only
    ever one-directional.

    Raises:
        CvxDataError: If ``x`` does not fit into ``(rows, cols)``.
    """
    # I had no luck with ndarray.resize()
    (n, m) = np.shape(x)

    if n > rows or m > cols:
        raise CvxDataError(f"Matrix of shape {(n, m)} does not fit a problem built for {(rows, cols)}")  # noqa: TRY003

    z = np.zeros((rows, cols))
    z[:n, :m] = x
    return z
