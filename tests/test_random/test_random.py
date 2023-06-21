# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from cvx.random import rand_cov


def test_rand_cov():
    a = rand_cov(5)
    return np.all(np.linalg.eigvals(a) > 0)
