# -*- coding: utf-8 -*-
"""PCA analysis with numpy"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import numpy as np

PCA = namedtuple(
    "PCA",
    [
        "explained_variance",
        "factors",
        "exposure",
        "cov",
        "systematic_returns",
        "idiosyncratic_returns",
        "idiosyncratic_risk",
    ],
)


def pca(returns, n_components=10):
    """
    Compute the first n principal components for a return matrix without sklearn

    Args:
        returns: DataFrame of prices
        n_components: Number of compoFnents
    """

    if n_components > returns.shape[1]:
        raise ValueError("The number of components cannot exceed the number of assets")

    # compute the principal components without sklearn
    # 1. compute the correlation
    cov = np.cov(returns.T)
    # 2. compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # 3. sort the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])
    # 4. compute the factors
    factors = returns @ eigenvectors[:, :n_components]
    # 5. compute the exposure
    exposure = np.transpose(eigenvectors[:, :n_components])

    return PCA(
        explained_variance=eigenvalues[:n_components] / (np.sum(eigenvalues)),
        factors=factors,
        exposure=exposure,
        cov=np.cov(factors.T),
        systematic_returns=factors @ exposure,
        idiosyncratic_returns=returns - factors @ exposure,
        idiosyncratic_risk=np.std(returns - factors @ exposure, axis=0),
    )


@dataclass
class PPCA:
    n_components: int = 0
    returns: np.ndarray = np.array([])

    def __post_init__(self):
        if self.n_components > self.returns.shape[1]:
            raise ValueError(
                "The number of components cannot exceed the number of assets"
            )

        # compute the principal components without sklearn
        # 1. compute the correlation
        cov = np.cov(self.returns.T)
        # 2. compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # 3. sort the eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = np.real(eigenvalues[idx])
        eigenvectors = np.real(eigenvectors[:, idx])
        # 4. compute the factors
        self.factors = self.returns @ eigenvectors[:, : self.n_components]

        self.exposure = np.transpose(eigenvectors[:, : self.n_components])

    @property
    def cov(self):
        return np.cov(self.factors.T)

    @property
    def systematic_returns(self):
        return self.factors @ self.exposure

    @property
    def idiosyncratic_returns(self):
        return self.returns - self.systematic_returns

    @property
    def idiosyncratic_risk(self):
        return np.std(self.idiosyncratic_returns, axis=0)
