# -*- coding: utf-8 -*-
"""PCA analysis
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA

PCA = namedtuple(
    "PCA",
    ["explained_variance", "factors", "exposure", "cov", "systematic", "idiosyncratic"],
)


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
        explained_variance=sklearn_pca.explained_variance_ratio_,
        factors=factors,
        exposure=pd.DataFrame(data=exposure, columns=returns.columns),
        cov=factors.cov(),
        systematic=pd.DataFrame(
            data=factors.values @ exposure, index=returns.index, columns=returns.columns
        ),
        idiosyncratic=pd.DataFrame(
            data=returns.values - factors.values @ exposure,
            index=returns.index,
            columns=returns.columns,
        ),
    )
