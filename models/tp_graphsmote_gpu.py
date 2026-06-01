"""
Optimized Topology-Preserving GraphSMOTE for GPU.
Pre-computes KNN and uses vectorized operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TPGraphSMOTEGPU(nn.Module):
    def __init__(self, embedding_dim, k_neighbors=5, beta=0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.beta = beta

        self.attribute_completion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        self.edge_decoder_weight = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim) * 0.01
        )

    def forward(self, node_embeddings, labels, adj, oversample_ratio=2.0):
        """
        GPU-optimized oversampling with vectorized operations.
        """
        device = node_embeddings.device
        num_nodes = node_embeddings.shape[0]

        fraud_mask = labels == 1
        fraud_indices = torch.where(fraud_mask)[0]
        fraud_embeddings = node_embeddings[fraud_mask]
        num_fraud = fraud_embeddings.shape[0]

        if num_fraud == 0 or oversample_ratio <= 1.0:
            return node_embeddings, labels, adj, torch.tensor(0.0, device=device)

        num_to_generate = int(num_fraud * (oversample_ratio - 1))
        if num_to_generate <= 0:
            return node_embeddings, labels, adj, torch.tensor(0.0, device=device)

        # GPU-based KNN using pairwise distances
        k = min(self.k_neighbors, num_fraud - 1)
        if k <= 0:
            return node_embeddings, labels, adj, torch.tensor(0.0, device=device)

        # Compute pairwise distances on GPU
        with torch.no_grad():
            dists = torch.cdist(fraud_embeddings, fraud_embeddings)
            dists.fill_diagonal_(float('inf'))
            _, knn_indices = dists.topk(k, dim=1, largest=False)

        # Vectorized synthetic node generation
        base_idx = torch.randint(0, num_fraud, (num_to_generate,), device=device)
        neighbor_choice = torch.randint(0, k, (num_to_generate,), device=device)
        neighbor_idx = knn_indices[base_idx, neighbor_choice]

        base_embs = fraud_embeddings[base_idx]
        neighbor_embs = fraud_embeddings[neighbor_idx]
        deltas = torch.rand(num_to_generate, 1, device=device)

        synthetic_raw = (1 - deltas) * base_embs + deltas * neighbor_embs
        synthetic = self.attribute_completion(synthetic_raw)

        new_embeddings = torch.cat([node_embeddings, synthetic], dim=0)
        new_labels = torch.cat([
            labels,
            torch.ones(num_to_generate, dtype=labels.dtype, device=device)
        ])

        # Laplacian loss: synthetic nodes should be close to their parents
        parent_embs = node_embeddings[fraud_indices[base_idx]]
        diff = synthetic - parent_embs
        lap_loss = torch.mean(torch.sum(diff ** 2, dim=1))

        return new_embeddings, new_labels, adj, lap_loss
