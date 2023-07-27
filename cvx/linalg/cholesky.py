# -*- coding: utf-8 -*-
"""Cholesky decomposition with numpy"""
from __future__ import annotations

import numpy as np


def cholesky(cov: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Compute the cholesky decomposition of a covariance matrix"""
    # upper triangular part of the cholesky decomposition
    # np.linalg.cholesky(cov) is the lower triangular part
    return np.transpose(np.linalg.cholesky(cov))
