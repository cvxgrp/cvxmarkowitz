"""Tests for sample-covariance risk model including robust variants."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import cholesky
from cvx.markowitz.builder import CvxError
from cvx.markowitz.names import DataNames as D
from cvx.markowitz.risk import SampleCovariance


def test_sample():
    """Plain sample risk for 2 assets reproduces expected volatility."""
    riskmodel = SampleCovariance(assets=2)
    riskmodel.update(
        **{
            D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.VOLA_UNCERTAINTY: np.zeros(2),
        }
    )

    vola = riskmodel.estimate({D.WEIGHTS: np.array([1.0, 1.0]), D._ABS: np.array([1.0, 1.0])}).value
    np.testing.assert_almost_equal(vola, 2.0)


def test_sample_large():
    """Estimate risk for a 4-asset setup where only first two assets are active.

    Verifies that padding with zeros (inactive assets) preserves the 2-asset
    reference result for volatility.
    """
    riskmodel = SampleCovariance(assets=4)
    riskmodel.update(
        **{
            D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.VOLA_UNCERTAINTY: np.zeros(2),
        }
    )
    vola = riskmodel.estimate(
        {
            D.WEIGHTS: np.array([1.0, 1.0, 0.0, 0.0]),
            D._ABS: np.array([1.0, 1.0, 0.0, 0.0]),
        }
    ).value

    np.testing.assert_almost_equal(vola, 2.0)


def test_robust_sample():
    """Compute robust risk for 2 assets with per-asset uncertainty.

    Confirms that the robust term based on absolute weights contributes to
    the resulting norm as expected.
    """
    riskmodel = SampleCovariance(assets=2)
    riskmodel.update(
        **{
            D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.VOLA_UNCERTAINTY: np.array([0.1, 0.2]),
        }
    )

    # Note: dummy should be abs(weights)
    vola = riskmodel.estimate({D.WEIGHTS: np.array([1.0, -1.0]), D._ABS: np.array([1.0, 1.0])}).value
    np.testing.assert_almost_equal(vola, np.sqrt(2.09))


def test_robust_sample_large():
    """Robust risk with 4 assets where only two assets are active.

    Ensures the robust volatility matches the 2-asset benchmark.
    """
    riskmodel = SampleCovariance(assets=4)
    riskmodel.update(
        **{
            D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
            D.LOWER_BOUND_ASSETS: np.zeros(2),
            D.UPPER_BOUND_ASSETS: np.ones(2),
            D.VOLA_UNCERTAINTY: np.array([0.1, 0.2]),
        }
    )

    vola = riskmodel.estimate(
        {
            D.WEIGHTS: np.array([1.0, -1.0, 0.0, 0.0]),
            D._ABS: np.array([1.0, 1.0, 0.0, 0.0]),
        }
    ).value

    np.testing.assert_almost_equal(vola, np.sqrt(2.09))


def test_mismatch():
    """Updating with mismatched vector lengths should raise CvxError."""
    riskmodel = SampleCovariance(assets=4)

    with pytest.raises(CvxError):
        riskmodel.update(
            **{
                D.CHOLESKY: cholesky(np.array([[1.0, 0.5], [0.5, 2.0]])),
                D.LOWER_BOUND_ASSETS: np.zeros(1),
                D.UPPER_BOUND_ASSETS: np.ones(1),
                D.VOLA_UNCERTAINTY: np.array([0.1]),
            }
        )
