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
"""Cholesky decomposition with numpy"""

from __future__ import annotations

import numpy as np

from .types import Matrix


def cholesky(cov: Matrix) -> Matrix:
    """Compute the cholesky decomposition of a covariance matrix"""
    # upper triangular part of the cholesky decomposition
    # np.linalg.cholesky(cov) is the lower triangular part
    return np.transpose(np.linalg.cholesky(cov))
