# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import valid


def test_valid():
    a = np.array([[np.NaN, np.NaN], [np.NaN, 4]])
    v, mat = valid(a)

    assert np.allclose(mat, np.array([[4]]))
    assert np.allclose(v, np.array([False, True]))


def test_invalid():
    a = np.zeros((3, 2))
    with pytest.raises(AssertionError):
        valid(a)
