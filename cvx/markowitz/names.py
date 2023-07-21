# -*- coding: utf-8 -*-
# see https://stackoverflow.com/a/54950492/1695486


class DataNames:
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
    #
    # @classmethod
    # def required_constraints(cls) -> list[ConstraintName]:
    #     """
    #     Return the required constraints
    #     """
    #     return [cls.BUDGET]
    #
    # @classmethod
    # def try_from_string(cls, string: str) -> ConstraintName | str:
    #     """
    #     Try to convert a string to a constraint name, otherwise return the string
    #     """
    #     try:
    #         return cls(string.lower())
    #     except (AttributeError, ValueError):
    #         return string
    #
    # @classmethod
    # def validate_constraints(cls, problem_constraints: list[ConstraintName | str]):
    #     """
    #     Validate the presence of all required constraints
    #     """
    #     problem_constraints = [cls.try_from_string(c) for c in problem_constraints]
    #
    #     required_constraints = set(cls.required_constraints())
    #     missing_constraints = required_constraints - set(problem_constraints)
    #     if missing_constraints:
    #         raise CvxError(
    #             f"Missing required constraints: {[c.name for c in missing_constraints]}"
    #         )
