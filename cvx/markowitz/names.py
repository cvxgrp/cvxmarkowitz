# see https://stackoverflow.com/a/54950492/1695486


class DataNames:
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


class ParameterName:
    SIGMA_MAX = "sigma_max"
