from os import PathLike
from typing import Dict, Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

File = Union[str, bytes, PathLike]
Parameter = Dict[str, cp.Parameter]
Variables = Dict[str, cp.Variable]
Expressions = Dict[str, cp.Expression]
Constraints = Dict[str, cp.Constraint]

Matrix: TypeAlias = npt.NDArray[np.float64]
