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
"""PCA analysis with numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Matrix


@dataclass
class PCA:
    """Principal component analysis computed with NumPy only."""

    returns: Matrix
    n_components: int = 0

    def __post_init__(self) -> None:
        """Validate inputs and compute factors, exposures, and eigenvalues."""
        if self.n_components > self.returns.shape[1]:
            raise ValueError("The number of components cannot exceed the number of assets")  # noqa: TRY003

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
        """Proportion of total variance explained by the retained components."""
        return np.array(self.eigenvalues[: self.n_components] / np.sum(self.eigenvalues))

    @property
    def cov(self) -> Matrix:
        """Covariance matrix of retained factors."""
        return np.atleast_2d(np.cov(self.factors.T))

    @property
    def systematic_returns(self) -> Matrix:
        """Portion of returns explained by the PCA factors (F E^T)."""
        return np.array(self.factors @ self.exposure)

    @property
    def idiosyncratic_returns(self) -> Matrix:
        """Residual returns after removing the systematic (factor) component."""
        return self.returns - self.systematic_returns

    @property
    def idiosyncratic_vola(self) -> Matrix:
        """Per-asset standard deviation of idiosyncratic (residual) returns."""
        return np.array(np.std(self.idiosyncratic_returns, axis=0))
