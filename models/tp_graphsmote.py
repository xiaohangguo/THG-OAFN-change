"""
Topology-Preserving GraphSMOTE (TP-GraphSMOTE)
Adds Laplacian regularization to preserve fraud cluster topology
during oversampling. Core improvement over vanilla GraphSMOTE.

L_total = L_cls + beta * L_laplacian
L_laplacian = sum_{(i,j) in E} ||h_i - h_j||^2 * A_ij
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


class TPGraphSMOTE(nn.Module):
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

    def forward(self, node_embeddings, labels, adj_matrix, oversample_ratio=2.0):
        """
        Returns:
            new_embeddings, new_labels, new_adj, laplacian_loss
        """
        device = node_embeddings.device
        num_nodes = node_embeddings.shape[0]

        fraud_mask = labels == 1
        fraud_indices = torch.where(fraud_mask)[0]
        fraud_embeddings = node_embeddings[fraud_mask]
        num_fraud = fraud_embeddings.shape[0]

        if num_fraud == 0 or oversample_ratio <= 1.0:
            zero_loss = torch.tensor(0.0, device=device)
            return node_embeddings, labels, adj_matrix, zero_loss

        num_to_generate = int(num_fraud * (oversample_ratio - 1))
        if num_to_generate <= 0:
            zero_loss = torch.tensor(0.0, device=device)
            return node_embeddings, labels, adj_matrix, zero_loss

        # KNN in embedding space
        fraud_np = fraud_embeddings.detach().cpu().numpy()
        k = min(self.k_neighbors, num_fraud)
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(fraud_np)

        new_embeddings_list = []
        parent_indices = []

        for i in range(num_to_generate):
            idx = np.random.randint(0, num_fraud)
            base = fraud_embeddings[idx:idx + 1]

            _, indices = knn.kneighbors(
                base.detach().cpu().numpy(), n_neighbors=min(2, num_fraud)
            )
            neighbor_idx = indices[0, 1] if indices.shape[1] > 1 else indices[0, 0]
            neighbor = fraud_embeddings[neighbor_idx:neighbor_idx + 1]

            delta = torch.rand(1, device=device)
            synthetic = (1 - delta) * base + delta * neighbor
            synthetic = self.attribute_completion(synthetic)

            new_embeddings_list.append(synthetic)
            parent_indices.append(fraud_indices[idx].item())

        new_fraud_embs = torch.cat(new_embeddings_list, dim=0)
        new_embeddings = torch.cat([node_embeddings, new_fraud_embs], dim=0)

        new_labels = torch.cat([
            labels,
            torch.ones(num_to_generate, dtype=labels.dtype, device=device)
        ])

        # Edge generation for synthetic nodes
        new_adj = self._generate_edges(new_embeddings, adj_matrix, num_nodes, num_to_generate)

        # Laplacian regularization
        lap_loss = self._laplacian_loss(
            new_embeddings, new_adj, num_nodes, num_to_generate, parent_indices
        )

        return new_embeddings, new_labels, new_adj, lap_loss

    def _generate_edges(self, embeddings, old_adj, num_old, num_new):
        device = embeddings.device
        total = num_old + num_new

        new_adj = torch.zeros(total, total, device=device)
        new_adj[:num_old, :num_old] = old_adj

        new_node_embs = embeddings[num_old:]
        scores = torch.sigmoid(
            torch.mm(
                torch.mm(new_node_embs, self.edge_decoder_weight),
                embeddings[:num_old].t()
            )
        )
        edges = (scores > 0.5).float()
        new_adj[num_old:, :num_old] = edges
        new_adj[:num_old, num_old:] = edges.t()

        return new_adj

    def _laplacian_loss(self, embeddings, adj, num_old, num_new, parent_indices):
        """
        Laplacian smoothness: synthetic nodes should be close to their
        graph neighbors in embedding space.
        """
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        count = 0

        for k in range(num_new):
            syn_idx = num_old + k
            syn_emb = embeddings[syn_idx]

            neighbors = torch.where(adj[syn_idx, :num_old] > 0)[0]
            if len(neighbors) == 0:
                parent = parent_indices[k]
                diff = syn_emb - embeddings[parent]
                loss = loss + torch.sum(diff ** 2)
                count += 1
            else:
                cap = min(5, len(neighbors))
                for nb in neighbors[:cap]:
                    diff = syn_emb - embeddings[nb]
                    loss = loss + torch.sum(diff ** 2) * adj[syn_idx, nb]
                    count += 1

        if count > 0:
            loss = loss / count

        return loss
