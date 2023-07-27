# -*- coding: utf-8 -*-
import numpy as np


def rand_cov(n: int) -> np.typing.NDArray[np.float64]:
    a = np.random.randn(n, n)
    return np.transpose(a) @ (a)
