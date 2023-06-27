# -*- coding: utf-8 -*-
"""PCA analysis with scikit-learn
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA

from cvx.linalg.pca import PCA


def pca(returns, n_components=10):
    """
    Compute the first n principal components for a return matrix

    Args:
        returns: DataFrame of prices
        n_components: Number of components
    """

    # USING SKLEARN. Let's look at the first n components
    sklearn_pca = sklearnPCA(n_components=n_components)
    sklearn_pca.fit_transform(returns)

    exposure = sklearn_pca.components_
    factors = returns @ np.transpose(exposure)

    return PCA(
        asset_names=returns.columns,
        factor_names=factors.columns,
        explained_variance=sklearn_pca.explained_variance_ratio_,
        factors=factors,
        exposure=pd.DataFrame(data=exposure, columns=returns.columns),
        cov=factors.cov(),
        # systematic_returns=factors @ exposure,
        systematic_returns=pd.DataFrame(
            data=factors.values @ exposure, index=returns.index, columns=returns.columns
        ),
        idiosyncratic_returns=pd.DataFrame(
            data=returns.values - factors.values @ exposure,
            index=returns.index,
            columns=returns.columns,
        ),
    )
