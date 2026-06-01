"""
GPU-compatible graph utilities.
Replaces DGL message passing with PyTorch sparse matrix operations
so everything runs on CUDA without needing DGL CUDA builds.
"""
import torch
import torch.nn.functional as F
import numpy as np


def build_sparse_adj(edge_index, num_nodes, edge_types=None, normalize=True):
    """
    Build a normalized sparse adjacency matrix from edge index.

    Args:
        edge_index: (2, E) numpy array or tensor
        num_nodes: number of nodes
        edge_types: optional (E,) array of edge types
        normalize: if True, apply symmetric normalization D^{-1/2} A D^{-1/2}

    Returns:
        adj: sparse FloatTensor (N, N)
        adj_per_type: dict mapping edge_type -> sparse adj (if edge_types provided)
    """
    if isinstance(edge_index, np.ndarray):
        src = torch.LongTensor(edge_index[0])
        dst = torch.LongTensor(edge_index[1])
    else:
        src = edge_index[0].long()
        dst = edge_index[1].long()

    # Add self-loops
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    src_all = torch.cat([src, self_loops])
    dst_all = torch.cat([dst, self_loops])

    values = torch.ones(src_all.shape[0], dtype=torch.float32)
    indices = torch.stack([src_all, dst_all])

    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    adj = adj.coalesce()

    if normalize:
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        vals = adj.values()
        row_idx = adj.indices()[0]
        col_idx = adj.indices()[1]
        new_vals = vals * deg_inv_sqrt[row_idx] * deg_inv_sqrt[col_idx]
        adj = torch.sparse_coo_tensor(adj.indices(), new_vals, (num_nodes, num_nodes))
        adj = adj.coalesce()

    # Per-type adjacency matrices
    adj_per_type = None
    if edge_types is not None:
        if isinstance(edge_types, np.ndarray):
            edge_types_t = torch.LongTensor(edge_types)
        else:
            edge_types_t = edge_types.long()

        unique_types = torch.unique(edge_types_t)
        adj_per_type = {}
        for etype in unique_types.tolist():
            mask = edge_types_t == etype
            if isinstance(edge_index, np.ndarray):
                e_src = torch.LongTensor(edge_index[0][mask.numpy() if isinstance(mask, torch.Tensor) else mask])
                e_dst = torch.LongTensor(edge_index[1][mask.numpy() if isinstance(mask, torch.Tensor) else mask])
            else:
                e_src = edge_index[0][mask]
                e_dst = edge_index[1][mask]

            e_vals = torch.ones(e_src.shape[0], dtype=torch.float32)
            e_indices = torch.stack([e_src, e_dst])
            e_adj = torch.sparse_coo_tensor(e_indices, e_vals, (num_nodes, num_nodes))
            e_adj = e_adj.coalesce()

            if normalize:
                e_deg = torch.sparse.sum(e_adj, dim=1).to_dense() + 1e-8
                e_deg_inv = 1.0 / e_deg
                e_row = e_adj.indices()[0]
                e_new_vals = e_adj.values() * e_deg_inv[e_row]
                e_adj = torch.sparse_coo_tensor(e_adj.indices(), e_new_vals, (num_nodes, num_nodes))
                e_adj = e_adj.coalesce()

            adj_per_type[etype] = e_adj

    return adj, adj_per_type


def sparse_message_passing(adj, h):
    """
    Perform message passing using sparse matrix multiplication.
    Equivalent to: for each node, aggregate neighbor features via mean.

    Args:
        adj: normalized sparse adjacency matrix (N, N) on same device as h
        h: node features (N, D)

    Returns:
        h_agg: aggregated features (N, D)
    """
    return torch.sparse.mm(adj, h)
