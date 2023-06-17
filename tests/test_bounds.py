# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cvx.markowitz.bounds import Bounds


def test_bounds():
    b = Bounds(upper=1, lower=0)
    assert b.upper == 1
    assert b.lower == 0

    with pytest.raises(FrozenInstanceError):
        b.upper = 2

    # b.upper = 2
    # b.upper = 3