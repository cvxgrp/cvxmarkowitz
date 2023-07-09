# -*- coding: utf-8 -*-
import numpy as np


def rand_cov(n):
    a = np.random.randn(n, n)
    return np.transpose(a) @ a
