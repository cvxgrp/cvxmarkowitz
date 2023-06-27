# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def cholesky(cov):
    # upper triangular part of the cholesky decomposition
    # np.linalg.cholesky(cov) is the lower triangular part
    return np.transpose(np.linalg.cholesky(cov))
