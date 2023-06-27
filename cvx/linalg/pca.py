# -*- coding: utf-8 -*-
"""PCA analysis
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd


PCA = namedtuple(
    "PCA",
    [
        "asset_names",
        "factor_names",
        "explained_variance",
        "factors",
        "exposure",
        "cov",
        "systematic_returns",
        "idiosyncratic_returns",
    ],
)


def pca(returns, n_components=10):
    """
    Compute the first n principal components for a return matrix without sklearn

    Args:
        returns: DataFrame of prices
        n_components: Number of compoFnents
    """

    if n_components > len(returns.columns):
        raise ValueError("The number of components cannot exceed the number of assets")

    # compute the principal components without sklearn
    # 1. compute the correlation
    corr = returns.cov()
    # 2. compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(corr.values)
    # 3. sort the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])
    # 4. compute the factors
    factors = returns @ eigenvectors[:, :n_components]
    # 5. compute the exposure
    exposure = pd.DataFrame(
        data=np.transpose(eigenvectors[:, :n_components]),
        columns=returns.columns,
        index=factors.columns,
    )

    return PCA(
        asset_names=returns.columns,
        factor_names=factors.columns,
        explained_variance=pd.Series(
            data=eigenvalues[:n_components] / (np.sum(eigenvalues))
        ),
        factors=factors,
        exposure=exposure,
        cov=factors.cov(),
        systematic_returns=factors @ exposure,
        idiosyncratic_returns=returns - factors @ exposure,
    )
