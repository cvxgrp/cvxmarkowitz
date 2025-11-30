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
# see https://stackoverflow.com/a/54950492/1695486

"""Centralized string constants used across the Markowitz package.

These names index data dictionaries, model maps, constraints, and parameters.
Keeping them in one place avoids typos and improves discoverability.
"""


class DataNames:
    """Keys for data items passed into models and builders."""

    RETURNS = "returns"
    MU = "mu"
    MU_UNCERTAINTY = "mu_uncertainty"
    CHOLESKY = "chol"
    VOLA_UNCERTAINTY = "vola_uncertainty"
    LOWER_BOUND_ASSETS = "lower_assets"
    LOWER_BOUND_FACTORS = "lower_factors"
    UPPER_BOUND_ASSETS = "upper_assets"
    UPPER_BOUND_FACTORS = "upper_factors"
    EXPOSURE = "exposure"
    HOLDING_COSTS = "holding_costs"
    IDIOSYNCRATIC_VOLA = "idiosyncratic_vola"
    IDIOSYNCRATIC_VOLA_UNCERTAINTY = "idiosyncratic_vola_uncertainty"
    SYSTEMATIC_VOLA_UNCERTAINTY = "systematic_vola_uncertainty"
    FACTOR_WEIGHTS = "factor_weights"
    WEIGHTS = "weights"
    _ABS = "_abs"


class ModelName:
    """Keys for model components bundled inside a builder/problem."""

    RISK = "risk"
    RETURN = "return"
    BOUND_ASSETS = "bound_assets"
    BOUND_FACTORS = "bound_factors"


class ConstraintName:
    """Canonical names for constraints used in builders/problems."""

    BUDGET = "budget"
    CONCENTRATION = "concentration"
    LONG_ONLY = "long_only"
    LEVERAGE = "leverage"
    RISK = "risk"


class ParameterName:
    """Keys for tunable problem parameters."""

    SIGMA_MAX = "sigma_max"
    OMEGA = "omega"
    SIGMA_TARGET = "sigma_target"
