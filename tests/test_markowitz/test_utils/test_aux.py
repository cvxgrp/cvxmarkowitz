"""Tests for the fill utilities that pad vectors/matrices with zeros.

The padding is what lets one compiled problem serve a universe smaller than the
one it was built for, so these cover both directions: values kept and the tail
zeroed on the way up, and a `CvxDataError` -- never a silent truncation -- for
input that does not fit at all.
"""

import numpy as np
import pytest

from cvxmarkowitz import CvxDataError
from cvxmarkowitz.utils.fill import fill_matrix, fill_vector


def test_fill_vector():
    """fill_vector should retain provided values and zero-pad to requested length."""
    a = np.ones(2)
    assert fill_vector(num=3, x=a) == pytest.approx(np.array([1.0, 1.0, 0.0]))


def test_fill_matrix():
    """fill_matrix should embed the input block in the top-left and zero-fill the rest."""
    a = np.ones((2, 2))
    expected = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert fill_matrix(rows=3, cols=3, x=a) == pytest.approx(expected)


def test_fill_vector_exact_fit():
    """A vector of exactly the target length is returned unchanged."""
    a = np.array([1.0, 2.0])
    assert fill_vector(num=2, x=a) == pytest.approx(a)


def test_fill_matrix_exact_fit():
    """A matrix of exactly the target shape is returned unchanged."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert fill_matrix(rows=2, cols=2, x=a) == pytest.approx(a)


def test_fill_vector_too_long():
    """A vector longer than the target raises CvxDataError, not a bare ValueError.

    The compiled problem has no room for the extra entries, and truncating them
    would silently drop assets the caller asked about. numpy's own message
    ("could not broadcast input array") names neither the caller's mistake nor a
    class inside the CvxError tree the README promises.
    """
    with pytest.raises(CvxDataError, match="length 3 does not fit a problem built for 2"):
        fill_vector(num=2, x=np.ones(3))


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (2, 3),  # too many rows
        (3, 2),  # too many columns
        (2, 2),  # too many of both
    ],
)
def test_fill_matrix_too_large(rows, cols):
    """A matrix exceeding the target in either axis raises CvxDataError."""
    with pytest.raises(CvxDataError, match="does not fit a problem built for"):
        fill_matrix(rows=rows, cols=cols, x=np.ones((3, 3)))
