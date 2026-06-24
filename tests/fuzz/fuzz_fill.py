"""Fuzz the cvxmarkowitz padding helpers against arbitrary shapes.

``fill_vector`` and ``fill_matrix`` pad an input array into a larger zero array
of a target shape. They must never crash with an unexpected exception on
adversarial input — mismatched/negative target shapes should raise a clean
error (ValueError/IndexError), not blow up unexpectedly. This harness exercises
that contract with coverage-guided input.

Run locally:
    pip install atheris numpy
    python tests/fuzz/fuzz_fill.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the native dependencies OUTSIDE the instrumentation block so they
# load uninstrumented; only the first-party package under test is instrumented.
# cvxmarkowitz.types imports cvxpy (which pulls scipy), so those are pre-imported
# too even though the fill helpers themselves only use numpy.
import cvxpy  # noqa: F401  # pre-imported uninstrumented
import numpy as np
import scipy.sparse  # noqa: F401  # pre-imported uninstrumented

with atheris.instrument_imports():
    from cvxmarkowitz.utils.fill import fill_matrix, fill_vector

_ALLOWED = (ValueError, IndexError, TypeError)


def test_one_input(data: bytes) -> None:
    """Pad fuzzed vectors/matrices to fuzzed target shapes."""
    fdp = atheris.FuzzedDataProvider(data)

    n = fdp.ConsumeIntInRange(0, 8)
    vec = np.array([fdp.ConsumeFloat() for _ in range(n)], dtype=np.float64)
    with contextlib.suppress(_ALLOWED):
        fill_vector(vec, fdp.ConsumeIntInRange(0, 12))

    r = fdp.ConsumeIntInRange(0, 6)
    c = fdp.ConsumeIntInRange(0, 6)
    mat = np.array([fdp.ConsumeFloat() for _ in range(r * c)], dtype=np.float64).reshape(r, c)
    with contextlib.suppress(_ALLOWED):
        fill_matrix(mat, fdp.ConsumeIntInRange(0, 8), fdp.ConsumeIntInRange(0, 8))


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
