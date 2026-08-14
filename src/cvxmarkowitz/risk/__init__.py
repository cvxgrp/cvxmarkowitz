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
"""Risk model subpackage public API."""

from __future__ import annotations

from cvxmarkowitz.model import Model

from .cvar.cvar import CVar as CVar
from .factor.factor import FactorModel as FactorModel
from .sample.sample import SampleCovariance as SampleCovariance

__all__ = ["CVar", "FactorModel", "SampleCovariance", "default_risk_model"]


def default_risk_model(assets: int, factors: int | None) -> Model:
    """Return the risk model a `Builder` uses when the caller injects none.

    A `FactorModel` when `factors` is given, a `SampleCovariance` otherwise.

    This rule lives here rather than in `Builder` so that the risk package owns
    the choice among its own members: adding a further default is a change to
    this subpackage, not to the abstract base class every builder inherits
    from. `Builder` therefore imports this function instead of the concrete
    model classes.

    Note what this does *not* do: it cannot default to `CVar`, which needs
    `rows` and `alpha` that a builder does not carry. `CVar` stays an injected
    model -- pass `model={ModelName.RISK: CVar(...)}` to the builder, which
    skips this function entirely.

    Args:
        assets: Number of assets the model is sized for.
        factors: Number of factors, or None for a non-factor problem.

    Returns:
        A `Model` instance to register under `ModelName.RISK`.
    """
    if factors is None:
        return SampleCovariance(assets=assets)

    return FactorModel(assets=assets, factors=factors)
