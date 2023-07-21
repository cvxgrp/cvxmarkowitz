# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.linalg.random import rand_cov


def test_cholesky():
    a = rand_cov(10)
    u = cholesky(a)
    # test numpy arrays are equivalent
    assert np.allclose(a, np.transpose(u) @ u)


def test_cholesky_not_symmetric():
    a = np.random.randn(10, 10)
    with pytest.raises(np.linalg.LinAlgError):
        cholesky(a)


def test_cholesky_not_square():
    a = np.random.randn(12, 10)
    with pytest.raises(np.linalg.LinAlgError):
        cholesky(a)
