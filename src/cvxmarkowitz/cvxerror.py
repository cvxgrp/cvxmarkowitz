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
"""Custom exceptions used by the Markowitz package."""


class CvxError(Exception):
    """Base error for the package.

    Every error raised by cvxmarkowitz derives from this, so
    ``except CvxError`` continues to catch all of them. Prefer catching one
    of the subclasses below when the handling differs by failure mode.
    """


class CvxDataError(CvxError):
    """Input data is missing, or its shape disagrees with the model.

    Raised when required keyword data is absent, or when arrays that must
    agree in length or shape do not. Recoverable by supplying corrected
    input and retrying.
    """


class CvxSolverError(CvxError):
    """The solver returned a non-optimal status.

    Raised when a problem is infeasible, unbounded, or otherwise did not
    solve to optimality. Retrying with the same input will not help;
    another solver or a relaxed formulation might.
    """


class CvxTrustError(CvxError):
    """A trust boundary was crossed without explicit consent.

    Raised when :func:`~cvxmarkowitz.problem.deserialize` is called without
    ``trusted=True``. This is a security guard, not a transient failure —
    catching it to retry with ``trusted=True`` defeats its purpose.
    """
