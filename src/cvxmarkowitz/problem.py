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
"""The built problem container returned by :meth:`Builder.build`."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np

from cvxmarkowitz.cvxerror import CvxDataError, CvxSolverError
from cvxmarkowitz.model import Model
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.types import Matrix, Parameter, Variables


@dataclass(frozen=True)
class Problem:
    """Frozen container holding a built cvxpy problem and its named models."""

    problem: cp.Problem
    model: dict[str, Model] = field(default_factory=dict)

    def update(self, **kwargs: Matrix) -> None:
        """Overwrite the parameter values of every model, **in place**.

        This mutates the problem rather than returning a new one. `frozen=True`
        on this dataclass only stops attribute rebinding; the `model` mapping and
        the cvxpy Parameters it holds stay mutable, and that is deliberate --
        writing new values into the same compiled problem is exactly what lets
        cvxpy reuse its cached canonicalization across solves:

            problem = MinVar(assets=4).build()   # compile once
            for data in datasets:
                problem.update(**data)           # overwrite in place
                problem.solve()

        Consequently there is only ever one problem. Two `update` calls against
        the same object do not yield two independently parametrized problems --
        the second overwrites the first. Call `build()` again for that.

        The whole payload is validated against every model before the first
        value is written, so a rejected payload leaves the problem exactly as it
        was rather than half-overwritten. See `_validate`.

        Returns `None` (like `Model.update`) so the in-place semantics are
        visible at the call site.

        Raises:
            CvxDataError: If any model is missing data for one of its parameters,
                or if the models disagree about how large the universe is.
        """
        self._validate(**kwargs)

        for model in self.model.values():
            # It's tempting to operate without the models at this stage.
            # However, we would give up a lot of convenience. For example,
            # the models can be prepared to deal with data that has not
            # exactly the correct shape.
            model.update(**kwargs)

    def _validate(self, **kwargs: Matrix) -> None:
        """Check the payload against every model, writing nothing.

        Two passes, both over all models, both raising `CvxDataError`:

        1. every keyword each model declares is present, and
        2. the models agree on the size of each variable they describe.

        The second is not redundant with the shape checks inside the models.
        `Model.update` pads a short input up to the compiled size, so a payload
        that describes two assets to the risk model and four to the bounds
        solves without complaint -- and solves wrongly, because the padded tail
        carries no risk while the bounds leave it free, which the solver reads as
        two riskless assets. Nothing inside a single model can see that; only
        comparing the models can.

        Raises:
            CvxDataError: On a missing keyword, or on models that disagree about
                the size of a variable.
        """
        for name, model in self.model.items():
            # `Model.keywords`, not `model.data`: a model may consume a keyword
            # that `data` does not back (see `ExpectedReturns.keywords`), and
            # checking `data` alone let those through to a bare KeyError.
            for key in model.keywords:
                if key not in kwargs:
                    raise CvxDataError(f"Missing data for {key} in model {name}")  # noqa: TRY003

        claimed: dict[str, tuple[str, int]] = {}

        for name, model in self.model.items():
            for variable, size in model.dimensions(**kwargs):
                first_name, first_size = claimed.setdefault(variable, (name, size))

                if size != first_size:
                    raise CvxDataError(  # noqa: TRY003
                        f"Inconsistent size for {variable}: model {first_name} was given "
                        f"{first_size}, model {name} was given {size}"
                    )

    def solve(self, solver: str = cp.CLARABEL, **kwargs: Any) -> float:
        """Solve the problem."""
        value = self.problem.solve(solver=solver, **kwargs)

        if self.problem.status is not cp.OPTIMAL:
            raise CvxSolverError(f"Problem status is {self.problem.status}")  # noqa: TRY003

        return float(value)

    def get_problem_data(
        self,
        solver: str = cp.CLARABEL,
        gp: bool = False,
        enforce_dpp: bool = False,
        ignore_dpp: bool = False,
        verbose: bool = False,
        canon_backend: str | None = None,
        solver_opts: dict[str, Any] | None = None,
    ) -> Any:
        """Return the low-level data the solver would be handed for this problem.

        This forwards to :meth:`cvxpy.Problem.get_problem_data`, exposing the
        compiled form of the problem without solving it. Useful for inspecting
        the canonicalization or for driving a solver directly.

        Args:
            solver: The target solver to compile for.
            gp: Whether to parse the problem as a disciplined geometric program.
            enforce_dpp: Raise if the problem is not DPP-compliant.
            ignore_dpp: Treat the problem as non-DPP even if it is compliant.
            verbose: Print compilation progress.
            canon_backend: Canonicalization backend to use, or None for the default.
            solver_opts: Extra options forwarded to the solver.

        Returns:
            The ``(data, chain, inverse_data)`` triple produced by cvxpy.
        """
        return self.problem.get_problem_data(
            solver,
            gp=gp,
            enforce_dpp=enforce_dpp,
            ignore_dpp=ignore_dpp,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_opts=solver_opts,
        )

    @property
    def value(self) -> float:
        """Return the current objective value of the solved problem."""
        return float(self.problem.value)

    def is_dpp(self) -> bool:
        """Return True if the problem satisfies disciplined parameterized programming."""
        return bool(self.problem.is_dpp())

    @property
    def data(self) -> Generator[tuple[tuple[str, str], cp.Parameter]]:
        """Yield ``((model_name, param_key), parameter)`` pairs for all models."""
        for name, model in self.model.items():
            for key, value in model.data.items():
                yield (name, key), value

    @property
    def parameter(self) -> Parameter:
        """Return a mapping of parameter names to cvxpy Parameter objects."""
        return dict(self.problem.param_dict.items())

    @property
    def variables(self) -> Variables:
        """Return a mapping of variable names to cvxpy Variable objects."""
        return dict(self.problem.var_dict.items())

    @property
    def weights(self) -> Matrix:
        """Return the optimal asset weights as a numpy array."""
        return np.array(self.variables[D.WEIGHTS].value)

    @property
    def factor_weights(self) -> Matrix:
        """Return the optimal factor weights as a numpy array.

        Raises:
            CvxDataError: If the problem was built without a factor risk model,
                in which case there is no factor-weight variable.
        """
        try:
            return np.array(self.variables[D.FACTOR_WEIGHTS].value)
        except KeyError as err:
            raise CvxDataError(  # noqa: TRY003
                "No factor weights: this problem was built without 'factors'."
            ) from err
