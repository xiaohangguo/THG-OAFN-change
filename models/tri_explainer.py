"""
Three-Level Explainer (TriExplainer)
Provides feature-level, edge-level, and subgraph-level attribution
for fraud detection predictions. Consumes XAttention weights.
"""
import torch
import json
import numpy as np


class TriExplainer:
    """Post-hoc explainer that consumes XAttention weights."""

    def __init__(self, feature_names=None, top_k_features=5, top_k_edges=10,
                 subgraph_threshold=0.1):
        self.feature_names = feature_names
        self.top_k_features = top_k_features
        self.top_k_edges = top_k_edges
        self.subgraph_threshold = subgraph_threshold

    def explain_node(self, node_idx, explanation_dict, g, node_features,
                     predictions, probabilities):
        """
        Generate three-level explanation for a single node.

        Returns:
            dict with feature_attribution, edge_attribution, subgraph_attribution
        """
        result = {
            'node_idx': int(node_idx),
            'predicted_label': int(predictions[node_idx].item()),
            'fraud_probability': float(probabilities[node_idx, 1].item()),
        }

        result['feature_attribution'] = self._feature_attribution(
            explanation_dict, node_features[node_idx]
        )
        result['edge_attribution'] = self._edge_attribution(
            node_idx, explanation_dict, g
        )
        result['subgraph_attribution'] = self._subgraph_attribution(
            node_idx, g, probabilities
        )

        return result

    def _feature_attribution(self, explanation_dict, node_feature):
        """Gate weights indicate which feature dimensions matter most."""
        gate_weights = explanation_dict.get('gate_weights', None)
        if gate_weights is None:
            return []

        num_heads = len(gate_weights)
        feat_dim = node_feature.shape[0]
        head_dim = feat_dim // num_heads

        importance = torch.zeros(feat_dim)
        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            importance[start:end] = gate_weights[h].item() * torch.abs(node_feature[start:end])

        topk_vals, topk_idx = torch.topk(
            importance, min(self.top_k_features, len(importance))
        )

        attributions = []
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            name = (self.feature_names[idx]
                    if self.feature_names and idx < len(self.feature_names)
                    else f"dim_{idx}")
            attributions.append({
                'feature': name,
                'importance': val,
                'value': float(node_feature[idx])
            })
        return attributions

    def _edge_attribution(self, node_idx, explanation_dict, g):
        """Identify which neighbor connections contribute most."""
        predecessors = g.predecessors(node_idx).cpu().numpy().tolist()
        successors = g.successors(node_idx).cpu().numpy().tolist()
        neighbors = list(set(predecessors + successors))

        if not neighbors:
            return []

        rel_weights = explanation_dict.get('relation_weights', None)
        edge_types = g.edata.get('edge_type', None)

        edge_attrs = []
        src, dst = g.edges()

        for nb in neighbors[:self.top_k_edges]:
            attr = {'neighbor_idx': int(nb)}
            mask = ((src == node_idx) & (dst == nb)) | ((src == nb) & (dst == node_idx))
            if mask.any() and edge_types is not None:
                etype = edge_types[mask][0].item()
                attr['edge_type'] = int(etype)
                if rel_weights is not None and etype < len(rel_weights):
                    attr['relation_importance'] = float(rel_weights[etype].item())
            edge_attrs.append(attr)

        edge_attrs.sort(key=lambda x: x.get('relation_importance', 0), reverse=True)
        return edge_attrs[:self.top_k_edges]

    def _subgraph_attribution(self, node_idx, g, probabilities):
        """Extract minimal subgraph explaining the prediction via BFS."""
        visited = {int(node_idx)}
        frontier = [int(node_idx)]
        subgraph_nodes = [int(node_idx)]

        for hop in range(2):
            next_frontier = []
            for n in frontier:
                neighbors = g.successors(n).cpu().numpy().tolist()
                neighbors += g.predecessors(n).cpu().numpy().tolist()
                for nb in set(neighbors):
                    if nb not in visited:
                        visited.add(nb)
                        if probabilities[nb, 1].item() > self.subgraph_threshold:
                            subgraph_nodes.append(int(nb))
                            next_frontier.append(nb)
            frontier = next_frontier

        return {
            'center_node': int(node_idx),
            'subgraph_nodes': subgraph_nodes,
            'num_nodes': len(subgraph_nodes),
            'max_hops': 2,
        }

    def explain_batch(self, node_indices, explanation_dict, g, node_features,
                      predictions, probabilities):
        """Explain multiple nodes."""
        return [
            self.explain_node(idx, explanation_dict, g, node_features,
                              predictions, probabilities)
            for idx in node_indices
        ]

    def to_json(self, explanations, filepath):
        """Save explanations to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(explanations, f, indent=2, ensure_ascii=False)
