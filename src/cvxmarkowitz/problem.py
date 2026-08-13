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
"""The built problem container and its (de)serialization round-trip."""

from __future__ import annotations

import pickle  # nosec B403
from collections.abc import Generator
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import cvxpy as cp
import numpy as np

from cvxmarkowitz.cvxerror import CvxDataError, CvxSolverError, CvxTrustError
from cvxmarkowitz.model import Model
from cvxmarkowitz.names import DataNames as D
from cvxmarkowitz.types import File, Matrix, Parameter, Variables


def deserialize(
    problem_file: str | bytes | PathLike[str] | PathLike[bytes] | int,
    *,
    trusted: bool = False,
) -> Any:
    """Load a previously serialized Markowitz problem from disk.

    .. warning::

        This uses :func:`pickle.load`, which executes arbitrary code while
        unpickling. Only ever call this on files you produced yourself with
        :meth:`Problem.serialize`. Never deserialize a file received from an
        untrusted or unauthenticated source — doing so is equivalent to
        running that source's code on your machine.

    To make that trust boundary explicit, deserialization is opt-in: you must
    pass ``trusted=True`` to confirm the file is one you produced yourself.
    Calling without it raises :class:`~cvxmarkowitz.cvxerror.CvxError` rather
    than silently unpickling.

    Args:
        problem_file: Path to the pickle file created by `Problem.serialize`.
        trusted: Must be set to ``True`` to confirm the file originates from a
            trusted source. Defaults to ``False``, which refuses to load.

    Returns:
        The deserialized `Problem` instance.

    Raises:
        CvxError: If ``trusted`` is not explicitly set to ``True``.
    """
    if not trusted:
        raise CvxTrustError(  # noqa: TRY003
            "Refusing to deserialize: pickle.load executes arbitrary code. "
            "Pass trusted=True only for a file you produced yourself with "
            "Problem.serialize()."
        )
    # pickle is the intended format for round-tripping a built problem. The trust
    # boundary is guarded by the trusted flag above; the input is assumed to be a
    # self-produced serialize() file. Suppressions are on the call itself below.
    with open(problem_file, "rb") as infile:
        return pickle.load(infile)  # nosec B301  # noqa: S301


@dataclass(frozen=True)
class Problem:
    """Frozen container holding a built cvxpy problem and its named models."""

    problem: cp.Problem
    model: dict[str, Model] = field(default_factory=dict)

    def update(self, **kwargs: Matrix) -> Problem:
        """Update the problem."""
        for name, model in self.model.items():
            for key in model.data:
                if key not in kwargs:
                    raise CvxDataError(f"Missing data for {key} in model {name}")  # noqa: TRY003

            # It's tempting to operate without the models at this stage.
            # However, we would give up a lot of convenience. For example,
            # the models can be prepared to deal with data that has not
            # exactly the correct shape.
            model.update(**kwargs)

        return self

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
        """Return the optimal factor weights as a numpy array."""
        return np.array(self.variables[D.FACTOR_WEIGHTS].value)

    def serialize(self, problem_file: File) -> None:
        """Pickle this problem to disk for later reuse with `deserialize`.

        Call this before :meth:`solve`. Solving attaches a live solver handle to
        the underlying cvxpy problem, and that handle is not picklable — a
        solved problem raises ``TypeError: cannot pickle 'builtins.DefaultSolver'
        object`` here.

        Args:
            problem_file: Destination path for the pickle file.
        """
        with open(problem_file, "wb") as outfile:
            pickle.dump(self, outfile)
