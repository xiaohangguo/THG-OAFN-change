"""
GPU-compatible Explainable Multi-Layer Attention (XAttention).
Uses sparse matrix ops instead of DGL for full CUDA support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class XRelationFusion(nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        self.relation_weights = nn.Parameter(torch.zeros(num_relations))

    def forward(self, adj_per_type, explain=False):
        """
        Args:
            adj_per_type: dict {etype: sparse_adj} or None
        """
        weights = torch.sigmoid(self.relation_weights)
        if adj_per_type is None or len(adj_per_type) == 0:
            if explain:
                return None, weights
            return None

        fused = None
        for i, (etype, adj) in enumerate(adj_per_type.items()):
            if i < len(weights):
                w = weights[i]
            else:
                w = weights[-1]
            scaled = torch.sparse_coo_tensor(
                adj.indices(), adj.values() * w.item(), adj.shape
            ).coalesce()
            if fused is None:
                fused = scaled
            else:
                fused = (fused + scaled).coalesce()

        if explain:
            return fused, weights
        return fused


class XNeighborhoodFusion(nn.Module):
    def __init__(self, hidden_dim, num_hops=2, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads

        self.W_h = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)
        ])
        self.hop_weights = nn.Parameter(torch.zeros(num_hops))

    def forward(self, adj, node_features, explain=False):
        """
        Args:
            adj: sparse adjacency matrix (N, N)
            node_features: (N, D)
        """
        hop_embeddings = []
        current = node_features

        for k in range(self.num_hops):
            head_outputs = []
            for head_idx in range(self.num_heads):
                h_t = self.W_h[head_idx](current)
                h_agg = torch.sparse.mm(adj, h_t)
                head_outputs.append(h_agg)
            agg = torch.mean(torch.stack(head_outputs), dim=0)
            hop_embeddings.append(agg)
            current = agg

        hop_w = F.softmax(self.hop_weights, dim=0)
        h_fused = sum(hop_w[k] * hop_embeddings[k] for k in range(self.num_hops))

        if explain:
            return h_fused, {'hop_weights': hop_w.detach()}
        return h_fused


class XInformationPerception(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.W_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)
        ])
        self.gate_vector = nn.Parameter(torch.zeros(num_heads))

    def forward(self, h_multi, explain=False):
        gate_weights = F.softmax(self.gate_vector, dim=0)
        head_outputs = [self.W_heads[h](h_multi) for h in range(self.num_heads)]
        h_final = torch.cat(
            [gate_weights[h] * head_outputs[h] for h in range(self.num_heads)],
            dim=1
        )
        if explain:
            return h_final, {'gate_weights': gate_weights.detach()}
        return h_final


class XMultiLayerAttentionGPU(nn.Module):
    """GPU-compatible multi-layer attention using sparse ops."""
    def __init__(self, hidden_dim, num_relations=4, num_hops=2, num_heads=8):
        super().__init__()
        self.relation_fusion = XRelationFusion(num_relations)
        self.neighborhood_fusion = XNeighborhoodFusion(hidden_dim, num_hops, num_heads)
        self.information_perception = XInformationPerception(hidden_dim, num_heads)

    def forward(self, adj, node_features, adj_per_type=None, explain=False):
        """
        Args:
            adj: sparse adjacency matrix (N, N)
            node_features: (N, D)
            adj_per_type: dict {etype: sparse_adj} or None
            explain: whether to return attention weights
        """
        explanation = {}

        if adj_per_type is not None:
            if explain:
                fused_adj, rel_weights = self.relation_fusion(adj_per_type, explain=True)
                explanation['relation_weights'] = rel_weights
            else:
                fused_adj = self.relation_fusion(adj_per_type)
            use_adj = fused_adj if fused_adj is not None else adj
        else:
            use_adj = adj

        if explain:
            h_multi, neigh_info = self.neighborhood_fusion(use_adj, node_features, explain=True)
            explanation['neighborhood'] = neigh_info
        else:
            h_multi = self.neighborhood_fusion(use_adj, node_features)

        if explain:
            h_final, gate_info = self.information_perception(h_multi, explain=True)
            explanation['gate_weights'] = gate_info['gate_weights']
        else:
            h_final = self.information_perception(h_multi)

        if explain:
            return h_final, explanation
        return h_final
