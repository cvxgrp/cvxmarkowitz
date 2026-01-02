"""Tests for the `valid` helper that extracts a valid submatrix and mask.

These tests verify that non-finite rows/columns are removed consistently and
that non-quadratic inputs are rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxmarkowitz.linalg import valid


def test_valid():
    """Extract valid 1x1 submatrix from a 2x2 matrix with NaNs.

    Verifies that only the finite row/column remain and that the mask and
    resulting submatrix match expected values.
    """
    a = np.array([[np.nan, np.nan], [np.nan, 4]])
    v, mat = valid(a)

    assert np.allclose(mat, np.array([[4]]))
    assert np.allclose(v, np.array([False, True]))


def test_invalid():
    """Non-quadratic inputs should raise a ValueError."""
    a = np.zeros((3, 2))
    with pytest.raises(ValueError, match="Matrix must be quadratic"):
        valid(a)
