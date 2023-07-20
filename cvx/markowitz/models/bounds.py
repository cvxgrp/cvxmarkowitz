# -*- coding: utf-8 -*-
"""Bounds"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
import numpy as np

from cvx.markowitz.model import Model
from cvx.markowitz.names import VariableName as V


@dataclass(frozen=True)
class Bounds(Model):
    name: str = ""
    acting_on: str | V = "weights"

    def estimate(self, variables: Dict[str, cp.Variable]):
        """No estimation for bounds"""
        raise NotImplementedError("No estimation for bounds")

    def _f(self, str):
        return f"{str}_{self.name}"

    def __post_init__(self):
        self.data[self._f("lower")] = cp.Parameter(
            shape=self.assets,
            name=self._f("lower"),
            value=np.zeros(self.assets),
        )
        self.data[self._f("upper")] = cp.Parameter(
            shape=self.assets,
            name=self._f("upper"),
            value=np.ones(self.assets),
        )

    def _update(self, x):
        z = np.zeros(self.assets)
        z[: len(x)] = x
        return z

    def update(self, **kwargs):
        # lower = kwargs[self._f("lower")]
        self.data[self._f("lower")].value = self._update(
            kwargs[self._f("lower")]
        )  # np.zeros(self.assets)

        # upper = kwargs[self._f("upper")]
        self.data[self._f("upper")].value = self._update(kwargs[self._f("upper")])

    def constraints(
        self, variables: Dict[str | V, cp.Variable]
    ) -> Dict[str, cp.Expression]:
        return {
            f"lower bound {self.name}": variables[self.acting_on]
            >= self.data[self._f("lower")],
            f"upper bound {self.name}": variables[self.acting_on]
            <= self.data[self._f("upper")],
        }
