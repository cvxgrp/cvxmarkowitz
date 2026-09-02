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
"""Common type aliases used across the Markowitz package."""

from typing import TypeAlias

import cvxpy as cp
import numpy as np
import numpy.typing as npt

Parameter = dict[str, cp.Parameter]
Variables = dict[str, cp.Variable]
Constraints = dict[str, cp.Constraint]

Matrix: TypeAlias = npt.NDArray[np.float64]

# What `Model.dimensions` reports: (variable name, size) claims, one per input
# the model consumes. A tuple rather than a mapping because several inputs of
# one model speak about the same variable, and it is exactly their
# disagreement that `Problem.update` is looking for.
Dimensions: TypeAlias = tuple[tuple[str, int], ...]
