# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg.random import rand_cov


@pytest.mark.parametrize("size", [1, 5, 10, 20, 50, 100, 200, 500, 1000])
def test_rand_cov_eigvalsh(size):
    a = rand_cov(size)
    assert np.allclose(a, np.transpose(a))
    assert np.all(np.linalg.eigvalsh(a) > 0)


@pytest.mark.parametrize("size", [1, 5, 10, 20, 50, 100, 200, 500, 1000])
def test_rand_cov_eigvals(size):
    a = rand_cov(size)
    assert np.allclose(a, np.transpose(a))
    assert np.all(np.linalg.eigvals(a) > 0)
