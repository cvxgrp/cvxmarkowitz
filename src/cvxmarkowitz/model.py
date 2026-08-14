#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Abstract cp model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cvxpy as cp

from cvxmarkowitz.types import Constraints, Matrix, Parameter, Variables


@dataclass(frozen=True)
class Model(ABC):
    """Abstract base for every component a `Builder` assembles.

    Risk models are only part of it: `Bounds`, `ExpectedReturns`, `TradingCosts`
    and `HoldingCosts` derive from this too. A component owns the cvxpy
    Parameters it is built from and contributes an objective term (`estimate`),
    constraints (`constraints`), or -- as `Bounds` does -- only the latter.

    Attributes:
        assets: Number of entries the component's parameters are sized for.
        parameter: cvxpy Parameters the component holds that `data` does not
            back. Some are set once at construction (`TradingCosts` fixes the
            cost exponent this way); others are written by `update` from a
            keyword, which is the case that requires overriding `keywords`.
        data: cvxpy Parameters `update` fills from its keyword arguments, and
            the default source of `keywords`.
    """

    assets: int
    parameter: Parameter = field(default_factory=dict)
    data: Parameter = field(default_factory=dict)

    @property
    def keywords(self) -> tuple[str, ...]:
        """Return the keyword names this model's `update` consumes.

        `Problem.update` checks these against the keywords it was handed before
        any value is written, so that a missing one is reported as a
        `CvxDataError` rather than escaping as whatever the model's own
        `kwargs[...]` lookup happens to raise.

        The default is the keys of `data`, which is where a model registers the
        cvxpy Parameters it fills from keyword arguments. **Override it whenever
        `update` reads a keyword that `data` does not back** -- otherwise the
        check cannot see that keyword and a caller who omits it gets a bare
        `KeyError`, which is outside the `CvxError` tree the package promises.
        `ExpectedReturns` is the one such model today.

        Returns a tuple rather than a set so the key named in the error message
        follows the insertion order of `data` instead of set iteration order.
        """
        return tuple(self.data)

    @abstractmethod
    def estimate(self, variables: Variables) -> cp.Expression:
        """Return this component's objective contribution, given the variables.

        What the expression means is the component's own business: a risk model
        returns a risk measure (`FactorModel` and `SampleCovariance` a norm, not
        a variance; `CVar` a conditional value-at-risk), `ExpectedReturns` a
        robust expected return, the cost models a cost. A component that
        contributes no objective term at all raises `NotImplementedError` here
        -- see `Bounds`, which is pure constraints.
        """

    @abstractmethod
    def update(self, **kwargs: Matrix) -> None:
        """Write fresh values into this component's parameters, in place.

        Each implementation documents the keywords it consumes; `keywords` is
        what `Problem.update` checks those against before any value is written.
        """

    def constraints(self, variables: Variables) -> Constraints:  # noqa: ARG002  # base default ignores `variables`; name kept to match overrides (LSP)
        """Return this component's named constraints; none by default."""
        return {}
