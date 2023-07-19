# -*- coding: utf-8 -*-
import cvxpy as cp
import numpy as np

from cvx.markowitz.portfolios.utils import approx


def test_approx():
    row = np.zeros(5)
    row[4] = 1
    row[2] = 1

    weights = cp.Variable(5)

    d = dict(approx("xxx", row @ weights, 2.0, 0.1))

    # I want this test!
    x = {
        "xxx_approx_upper": row @ weights - 2.0 <= 0.1,
        "xxx_approx_lower": row @ weights - 2.0 >= -0.1,
    }

    assert set(d.keys()) == {"xxx_approx_upper", "xxx_approx_lower"}

    # print(d.values())
    # print(x.values())

    # for c1, c2 in zip(d.values(), x.values()):
    #    assert c1.value() == c2.value()

    weights.value = np.array([0.2, 0.3, -0.15, 0.1, 0.3])
    print(d.values())
    print(x.values())

    for c1, c2 in zip(d.values(), x.values()):
        # assert c1 == c2
        assert str(c1.tree_copy()) == str(c2.tree_copy())
