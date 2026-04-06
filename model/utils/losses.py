# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/10 14:21
@Author  : Linjie Wang
@FileName: losses.py
@Software: PyCharm
'''
import torch
import torch.nn.functional as F
from torch import nn

def compute_laplacian_loss(Z, A):
    """
    Compute graph Laplacian regularization loss.

    Parameters
    ----------
    Z : torch.Tensor
        Node embeddings of shape (N, D)
    A : torch.Tensor
        Adjacency matrix of shape (N, N)

    Returns
    -------
    Scalar Laplacian loss
    """
    D = torch.diag(torch.sum(A, dim=1))
    L = D - A # Graph Laplacian
    Z = F.normalize(Z, dim=-1)

    return torch.trace(torch.matmul(torch.matmul(Z.T, L), Z)) / Z.shape[0]

def transpose(x):
    """Transpose last two dimensions."""
    return x.transpose(-2, -1)

def normalize(*xs):
    """Normalize each input tensor along the last dimension."""
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def info_nce(query, positive_key, negative_keys=None, temperature=1., reduction='mean', negative_mode='unpaired', norm=True):
    """
    Compute InfoNCE contrastive loss.

    Parameters
    ----------
    query : (N, D)
    positive_key : (N, D)
    negative_keys : (M, D) or (N, M, D)
    temperature : float
    reduction : str
    negative_mode : str
        'unpaired' or 'paired'
    normalize : bool
        Whether to apply L2 normalization

    Returns
    -------
    loss : torch.Tensor
    """
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    if norm:
        query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.
    """

    def __init__(self, reduction='mean', negative_mode='unpaired', temperature=0.07):
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.temperature = temperature

    def forward(self, query, positive_key, negative_keys=None, norm=True):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode, norm=norm)

