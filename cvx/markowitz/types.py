# -*- coding: utf-8 -*-
from os import PathLike
from typing import Dict, Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt


class Types:
    File = Union[str, bytes, PathLike[str], PathLike[bytes], int]
    Parameter = Dict[str, cp.Parameter]
    Variables = Dict[str, cp.Variable]
    Expressions = Dict[str, cp.Expression]
    Matrix = npt.NDArray[np.float64]


# read: https://adamj.eu/tech/2021/05/11/python-type-hints-args-and-kwargs/
UpdateData = Types.Matrix
