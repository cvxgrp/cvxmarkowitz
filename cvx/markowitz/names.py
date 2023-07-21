# -*- coding: utf-8 -*-
# see https://stackoverflow.com/a/54950492/1695486


class DataNames:
    RETURNS = "returns"
    MU = "mu"
    CHOLESKY = "chol"
    VOLA_UNCERTAINTY = "vola_uncertainty"
    LOWER_BOUND_ASSETS = "lower_assets"
    UPPER_BOUND_ASSETS = "upper_assets"
    EXPOSURE = "exposure"
    HOLDING_COSTS = "holding_costs"


class VariableName:
    WEIGHTS = "weights"
    FACTOR_WEIGHTS = "factor_weights"
    _ABS = "_abs"


class ModelName:
    RISK = "risk"
    RETURN = "return"
    BOUND_ASSETS = "bound_assets"
    BOUND_FACTORS = "bound_factors"


class ConstraintName:
    BUDGET = "budget"
    CONCENTRATION = "concentration"
    LONG_ONLY = "long_only"
    LEVERAGE = "leverage"
    RISK = "risk"
