# -*- coding: utf-8 -*-
"""PCA analysis with numpy"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvx.linalg.types import Matrix


@dataclass
class PCA:
    returns: Matrix
    n_components: int = 0

    def __post_init__(self) -> None:
        if self.n_components > self.returns.shape[1]:
            raise ValueError(
                "The number of components cannot exceed the number of assets"
            )

        # compute the principal components without sklearn
        # 1. compute the correlation
        cov = np.cov(self.returns.T)
        cov = np.atleast_2d(cov)

        # 2. compute the eigenvalues and eigenvectors
        self.eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 3. sort the eigenvalues in descending order
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # 4. compute the factors
        self.factors = self.returns @ eigenvectors[:, : self.n_components]

        self.exposure = np.transpose(eigenvectors[:, : self.n_components])

    @property
    def explained_variance(self) -> Matrix:
        return np.array(
            self.eigenvalues[: self.n_components] / np.sum(self.eigenvalues)
        )

    @property
    def cov(self) -> Matrix:
        return np.atleast_2d(np.cov(self.factors.T))

    @property
    def systematic_returns(self) -> Matrix:
        return np.array(self.factors @ self.exposure)

    @property
    def idiosyncratic_returns(self) -> Matrix:
        return self.returns - self.systematic_returns

    @property
    def idiosyncratic_vola(self) -> Matrix:
        return np.array(np.std(self.idiosyncratic_returns, axis=0))
