# -*- coding: utf-8 -*-
"""
Bounds
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bounds:
    lower: float = -np.infty
    upper: float = +np.infty


if __name__ == "__main__":
    assets = ["AAPL", "MSFT"]
    lower_bound = {"AAPL": 0.0, "MSFT": 0.0}
    upper_bound = {"AAPL": 1.0, "MSFT": 1.0}

    bounds = {
        asset: Bounds(lower=lower_bound[asset], upper=upper_bound[asset])
        for asset in assets
    }

    print(bounds)
