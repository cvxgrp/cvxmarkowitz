# -*- coding: utf-8 -*-
import numpy as np

from cvx.markowitz.utils.aux import fill_matrix, fill_vector


def test_fill_vector():
    a = np.ones(2)
    np.allclose(fill_vector(num=3, x=a), np.array([1, 1, 0]))


def test_fill_matrix():
    a = np.ones((2, 2))
    np.allclose(
        fill_matrix(rows=3, cols=3, x=a), np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    )
