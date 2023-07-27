# -*- coding: utf-8 -*-
"""PCA analysis with numpy"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class PCA:
    returns: np.typing.NDArray[np.float64]
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
    def explained_variance(self) -> npt.NDArray[np.float64]:
        return self.eigenvalues[: self.n_components] / np.sum(self.eigenvalues)

    @property
    def cov(self) -> npt.NDArray[np.float64]:
        return np.atleast_2d(np.cov(self.factors.T))

    @property
    def systematic_returns(self) -> npt.NDArray[np.float64]:
        return self.factors @ self.exposure

    @property
    def idiosyncratic_returns(self) -> npt.NDArray[np.float64]:
        return self.returns - self.systematic_returns

    @property
    def idiosyncratic_vola(self) -> npt.NDArray[np.float64]:
        return np.std(self.idiosyncratic_returns, axis=0)
