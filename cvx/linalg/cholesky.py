# -*- coding: utf-8 -*-
from __future__ import annotations

import scipy as sc


def cholesky(cov):
    # upper triangular part of the cholesky decomposition
    # np.linalg.cholesky(cov) is the lower triangular part
    return sc.linalg.cholesky(cov)
