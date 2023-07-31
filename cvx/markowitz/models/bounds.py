"""Bounds"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz.model import Model
from cvx.markowitz.types import Expressions, Matrix, Variables
from cvx.markowitz.utils.aux import fill_vector


@dataclass(frozen=True)
class Bounds(Model):
    name: str = ""
    acting_on: str = "weights"

    def estimate(self, variables: Variables) -> cp.Expression:
        """No estimation for bounds"""
        raise NotImplementedError("No estimation for bounds")

    def _f(self, string: str) -> str:
        return f"{string}_{self.name}"

    def __post_init__(self) -> None:
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

    def update(self, **kwargs: Matrix) -> None:
        self.data[self._f("lower")].value = fill_vector(
            num=self.assets, x=kwargs[self._f("lower")]
        )
        self.data[self._f("upper")].value = fill_vector(
            num=self.assets, x=kwargs[self._f("upper")]
        )

    def constraints(self, variables: Variables) -> Expressions:
        return {
            f"lower bound {self.name}": variables[self.acting_on]
            >= self.data[self._f("lower")],
            f"upper bound {self.name}": variables[self.acting_on]
            <= self.data[self._f("upper")],
        }
