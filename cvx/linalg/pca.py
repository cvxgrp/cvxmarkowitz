# -*- coding: utf-8 -*-
"""PCA analysis with numpy"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass
class PCA:
    n_components: int = 0
    returns: np.ndarray = field(default_factory=np.array)

    def __post_init__(self):
        if self.n_components > self.returns.shape[1]:
            raise ValueError(
                "The number of components cannot exceed the number of assets"
            )

        # compute the principal components without sklearn
        # 1. compute the correlation
        cov = np.cov(self.returns.T)
        # 2. compute the eigenvalues and eigenvectors
        self.eigenvalues, eigenvectors = np.linalg.eig(cov)
        # 3. sort the eigenvalues in descending order
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = np.real(self.eigenvalues[idx])
        eigenvectors = np.real(eigenvectors[:, idx])
        # 4. compute the factors
        self.factors = self.returns @ eigenvectors[:, : self.n_components]

        self.exposure = np.transpose(eigenvectors[:, : self.n_components])

    @property
    def explained_variance(self):
        return self.eigenvalues[: self.n_components] / (np.sum(self.eigenvalues))

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
