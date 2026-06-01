"""
Explainable Multi-Layer Attention (XAttention)
Same architecture as original but exposes attention weights for explanation.
Three layers: Relation Fusion -> Neighborhood Fusion -> Information Perception
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class XRelationFusion(nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        self.relation_weights = nn.Parameter(torch.zeros(num_relations))

    def forward(self, adj_matrices, explain=False):
        weights = torch.sigmoid(self.relation_weights)
        if adj_matrices is None or len(adj_matrices) == 0:
            if explain:
                return None, weights
            return None

        if torch.is_tensor(adj_matrices[0]) and adj_matrices[0].is_sparse:
            fused = weights[0] * adj_matrices[0]
            for i in range(1, len(adj_matrices)):
                fused = fused + weights[i] * adj_matrices[i]
        else:
            fused = torch.zeros_like(adj_matrices[0])
            for i, adj in enumerate(adj_matrices):
                fused += weights[i] * adj

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

    def forward(self, g, node_features, explain=False):
        hop_embeddings = []
        current = node_features

        for k in range(self.num_hops):
            agg = self._aggregate(g, current)
            hop_embeddings.append(agg)
            current = agg

        hop_w = F.softmax(self.hop_weights, dim=0)
        h_fused = sum(hop_w[k] * hop_embeddings[k] for k in range(self.num_hops))

        if explain:
            return h_fused, {'hop_weights': hop_w.detach()}
        return h_fused

    def _aggregate(self, g, features):
        head_outputs = []
        for head_idx in range(self.num_heads):
            h_t = self.W_h[head_idx](features)
            with g.local_scope():
                g.ndata['h'] = h_t
                g.update_all(
                    lambda edges: {'m': edges.src['h']},
                    lambda nodes: {'h_agg': torch.mean(nodes.mailbox['m'], dim=1)}
                )
                h_agg = g.ndata.get('h_agg', torch.zeros_like(h_t))
            head_outputs.append(h_agg)
        return torch.mean(torch.stack(head_outputs), dim=0)


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


class XMultiLayerAttention(nn.Module):
    def __init__(self, hidden_dim, num_relations=4, num_hops=2, num_heads=8):
        super().__init__()
        self.relation_fusion = XRelationFusion(num_relations)
        self.neighborhood_fusion = XNeighborhoodFusion(hidden_dim, num_hops, num_heads)
        self.information_perception = XInformationPerception(hidden_dim, num_heads)

    def forward(self, g, node_features, adj_matrices=None, explain=False):
        explanation = {}

        if explain:
            fused_adj, rel_weights = self.relation_fusion(adj_matrices, explain=True)
            explanation['relation_weights'] = rel_weights
        else:
            fused_adj = self.relation_fusion(adj_matrices)

        if explain:
            h_multi, neigh_info = self.neighborhood_fusion(g, node_features, explain=True)
            explanation['hop_weights'] = neigh_info['hop_weights']
        else:
            h_multi = self.neighborhood_fusion(g, node_features)

        if explain:
            h_final, gate_info = self.information_perception(h_multi, explain=True)
            explanation['gate_weights'] = gate_info['gate_weights']
        else:
            h_final = self.information_perception(h_multi)

        if explain:
            return h_final, explanation
        return h_final
