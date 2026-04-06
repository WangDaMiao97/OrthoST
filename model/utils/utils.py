# -*- coding: utf-8 -*-
'''
@Time    : 2024/9/5 10:33
@Author  : Linjie Wang
@FileName: utils.py
@Software: PyCharm
'''
import gc
import random
import torch
import pandas as pd
import numpy as np
from sklearn import neighbors

def set_seed(seed):
    print(f"\n===== Seed {seed} =====")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def generate_spatial_graph(adata, rad_cutoff=None, k_cutoff=None, model='Radius'):
    """
    Construct a spatial adjacency graph based on spatial coordinates.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial coordinates in `adata.obsm['spatial']`.
    rad_cutoff : float, optional
        Radius threshold for neighbor search (used when model='Radius').
    k_cutoff : int, optional
        Number of nearest neighbors (used when model='KNN').
    model : str, default='Radius'
        Graph construction method: 'Radius' or 'KNN'.

    Returns
    -------
    A : np.ndarray
        Binary adjacency matrix of shape (N, N).
    dist_matrix : np.ndarray
        Distance matrix of shape (N, N), storing pairwise distances for edges.
    """
    assert (model in ['Radius', 'KNN'])
    # Extract spatial coordinates
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.columns = ['imagerow', 'imagecol']

    # Initialize adjacency and distance matrices
    num_cells = coor.shape[0]
    A = np.zeros((num_cells, num_cells))
    dist_matrix = np.zeros((num_cells, num_cells))
    # Build neighbor graph
    if model == 'Radius':
        nbrs = neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    if model == 'KNN':
        nbrs = neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)

    # Collect edge indices
    for i in range(num_cells):
        for neighbor_idx, dist in zip(indices[i], distances[i]):
            # Skip self-loop
            if neighbor_idx == i:
                continue
            # Assign edge
            A[i, neighbor_idx] = 1
            dist_matrix[i, neighbor_idx] = dist

    return A, dist_matrix

def generate_feature_graph(adata, k_cutoff=3, use_data = "X_pca"):
    """
    Construct a kNN graph based on feature embeddings.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing feature embeddings in `adata.obsm`.
    k : int, default=3
        Number of nearest neighbors.
    use_data : str, default="X_pca"
        Key in `adata.obsm` specifying which embedding to use.

    Returns
    -------
    A : np.ndarray
        Binary adjacency matrix of shape (N, N).
    """
    if use_data in adata.obsm.keys():
        embedding = adata.obsm[use_data]
    else:
        embedding = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    num_cells = adata.n_obs

    # Build kNN graph (k+1 includes self)
    nbrs = neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    A = np.zeros((num_cells, num_cells), dtype=np.float32)
    for i in range(num_cells):
        for neighbor_idx in indices[i]:
            if neighbor_idx == i:
                continue  # skip self-loop
            A[i, neighbor_idx] = 1.0
    return A

def square_distance(src, dst):
    """
    Compute pairwise squared Euclidean distance.

    Parameters
    ----------
    src : torch.Tensor
        Source points, shape (N, C)
    dst : torch.Tensor
        Target points, shape (M, C)

    Returns
    -------
    dist : torch.Tensor
        Pairwise squared distances, shape (N, M)
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist = -2 * torch.matmul(src, dst.T)
    dist += torch.sum(src ** 2, dim=1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=1).unsqueeze(0)

    # Numerical stability (avoid negative due to float error)
    dist = torch.clamp(dist, min=0.0)

    return dist
#
def query_ball_point(n_sample=None, location=None, quartile=0.):
    """
    Group neighboring points within a radius (ball query).

    Parameters
    ----------
    nsample : int
        Maximum number of neighbors per point.
    location : torch.Tensor
        Input coordinates, shape (N, C).
    quartile : float, default=0.0
        If > 0, adaptively determine radius based on distance distribution.

    Returns
    -------
    group_idx : torch.Tensor
        Neighbor indices, shape (N, n_sample+1)
    """
    device = location.device
    N, _ = location.shape

    # Compute full pairwise distance matrix
    sqrdists = square_distance(location, location)
    gc.collect()

    # adaptive radius selection via quantile
    k = int(sqrdists.numel() * quartile)
    radius = torch.topk(sqrdists.view(-1), k, largest=False)[0][-1].item()
    # Initialize output
    group_idx = torch.empty((N, n_sample+1), dtype=torch.long, device=device)
    # Process in chunks to reduce memory pressure
    for i in range(0, N, 2000):
        end = min(i + 2000, N)
        chunk_dist = sqrdists[i:end]
        # Sort neighbors by distance
        sorted_dist, sorted_idx = torch.sort(chunk_dist, dim=1)
        # Mask out points outside radius
        mask = sorted_dist > radius
        sorted_idx[mask] = N  # invalid index
        group_idx[i:end] = sorted_idx[:, : n_sample + 1]
    del sqrdists
    gc.collect()

    # Replace invalid indices with self-index
    group_first = group_idx[:, 0].view(N, 1).repeat([1, n_sample+1])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def query_random_point(num_samples = None, n_sample = 4):
    """
    Randomly sample points for each point.

    Parameters
    ----------
    num_samples : int
        Total number of points (N).
    n_sample : int, default=4
        Number of neighbors to sample for each point (excluding itself).

    Returns
    -------
    group_idx : torch.Tensor
        Sampled neighbor indices of shape (N, n_sample + 1),
        where the first column is the point itself.
    """
    # Generate random indices in the range [0, num_points)
    group_idx = torch.randint(0, num_samples, (num_samples, n_sample+1))
    # Ensure the first column is the point itself
    group_idx[:,0] = torch.arange(num_samples)

    return group_idx

