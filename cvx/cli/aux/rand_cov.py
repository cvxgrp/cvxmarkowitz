# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def rand_cov(n):
    a = np.random.randn(n, n)
    return np.transpose(a) @ a
