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
from typing import Any, Generator, Tuple

import cvxpy as cp


def approx(
    name: str, x: cp.Variable, target: Any, pm: cp.Expression
) -> Generator[Tuple[str, cp.Expression], None, None]:
    yield f"{name}_approx_upper", x - target <= pm
    yield f"{name}_approx_lower", x - target >= -pm
