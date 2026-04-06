# -*- coding: utf-8 -*-
'''
@Time    : 2025/6/18 20:17
@Author  : Linjie Wang
@FileName: EBF.py
@Software: PyCharm
'''

import numpy as np
from scipy.sparse import csr_matrix, diags
def compute_sigma(dist_matrix):
    """
    Estimate Gaussian kernel bandwidth (sigma) using nearest neighbor distances.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix (N, N)

    Returns
    -------
    sigma : float
        Estimated bandwidth
    """
    # Ignore zero distances (self-loops)
    masked = np.where(dist_matrix > 0, dist_matrix, np.inf)
    # Nearest neighbor distance per node
    nn_dist = np.min(masked, axis=1)
    # Robust mean (avoid inf)
    sigma = np.mean(nn_dist[np.isfinite(nn_dist)])

    return sigma
def gaussian_smoothing_pipeline(X, graph, dist_spatial, dist_feature, feat_weight=0.5):
    """
    Full pipeline for spatial-feature Gaussian smoothing.
    """
    # Sigma estimation
    sigma_s = compute_sigma(dist_spatial)
    sigma_f = compute_sigma(dist_feature)

    # Compute Weight Matrix
    rows, cols = np.where(graph > 0)
    d_s = dist_spatial[rows, cols]
    d_f = dist_feature[rows, cols]
    weight_matrix = np.exp(
        -((1 - feat_weight) * (d_s ** 2) / (2 * sigma_s ** 2)
          + feat_weight * (d_f ** 2) / (2 * sigma_f ** 2)
          )
    ) # Calculate combined Gaussian weights
    weight_matrix =  csr_matrix((weight_matrix, (rows, cols)), shape=graph.shape)

    # Normalization
    row_sums = weight_matrix.sum(axis=1).A1
    diag_add = diags(row_sums, 0, format='csr')
    weight_matrix = weight_matrix + diag_add
    zero_indices = np.where(row_sums == 0)[0]
    if len(zero_indices) > 0:
        # Create a diagonal matrix with 1.0 at isolated indices
        fix_diag = np.zeros(weight_matrix.shape[0])
        fix_diag[zero_indices] = 1.0
        weight_matrix = weight_matrix + diags(fix_diag, 0, format='csr')
    weight_matrix /= weight_matrix.sum(axis=1).reshape((-1, 1))

    X_dense = X.toarray() if hasattr(X, "toarray") else X

    return weight_matrix.dot(X_dense)