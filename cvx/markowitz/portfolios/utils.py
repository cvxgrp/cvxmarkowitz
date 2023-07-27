from typing import Any, Generator, Tuple

import cvxpy as cp


def approx(
    name: str, x: cp.Variable, target: Any, pm: cp.Expression
) -> Generator[Tuple[str, cp.Expression], None, None]:
    yield f"{name}_approx_upper", x - target <= pm
    yield f"{name}_approx_lower", x - target >= -pm
