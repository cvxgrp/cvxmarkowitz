# -*- coding: utf-8 -*-
"""Bounds"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .model import Model


@dataclass
class Bounds(Model):
    m: int = 0
    name: str = ""

    def estimate(self, weights, **kwargs):
        """No estimation for bounds"""
        raise NotImplementedError("No estimation for bounds")

    def _f(self, str):
        return f"{str}_{self.name}"

    def __post_init__(self):
        self.parameter[self._f("lower")] = cp.Parameter(
            shape=self.m,
            name="lower bound",
            value=np.zeros(self.m),
        )
        self.parameter[self._f("upper")] = cp.Parameter(
            shape=self.m,
            name="upper bound",
            value=np.ones(self.m),
        )

    def update(self, **kwargs):
        # lower = kwargs.get("lower", np.zeros(self.m))
        lower = kwargs[self._f("lower")]
        self.parameter[self._f("lower")].value = np.zeros(self.m)
        self.parameter[self._f("lower")].value[: len(lower)] = lower

        upper = kwargs[self._f("upper")]  # .get("upper", np.ones(self.m))
        self.parameter[self._f("upper")].value = np.zeros(self.m)
        self.parameter[self._f("upper")].value[
            : len(upper)
        ] = upper  # kwargs.get("upper", np.ones(m))

    def constraints(self, weights, **kwargs):
        return [
            weights >= self.parameter[self._f("lower")],
            weights <= self.parameter[self._f("upper")],
        ]
