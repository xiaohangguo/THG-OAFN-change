"""
TriExplainer v3 — adapted for TP-THGN v3 (sparse adj, no DGL).
Three-level attribution: feature-level, edge-level, subgraph-level.
Uses gradient-based feature attribution + relation weights from GatedGraphLayer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np


class TriExplainerV3:
    """Post-hoc 3-level explainer for TP-THGN v3 (sparse adj, no DGL)."""

    def __init__(self, model, feature_names=None, top_k_features=5, top_k_edges=10):
        self.model = model
        self.feature_names = feature_names
        self.top_k_features = top_k_features
        self.top_k_edges = top_k_edges

    def explain_node(self, node_idx, node_features, adj, adj_per_type, labels):
        """Generate 3-level explanation for a single node."""
        self.model.eval()

        # 1. Feature-level: gradient-based attribution
        feat_attr = self._feature_attribution(node_idx, node_features, adj, adj_per_type)

        # 2. Edge-level: relation weights from GatedGraphLayer
        edge_attr = self._edge_attribution(node_idx, adj_per_type)

        # 3. Subgraph-level: high-risk neighborhood
        with torch.no_grad():
            preds, probs, explanation = self.model.predict_with_explanation(
                adj, node_features, None, adj_per_type)

        subgraph_attr = self._subgraph_attribution(node_idx, adj_per_type, probs)

        return {
            'node_idx': int(node_idx),
            'predicted_label': int(preds[node_idx].item()),
            'fraud_probability': float(probs[node_idx, 1].item()),
            'true_label': int(labels[node_idx].item()) if labels is not None else None,
            'feature_attribution': feat_attr,
            'edge_attribution': edge_attr,
            'subgraph_attribution': subgraph_attr,
            'relation_weights': {
                'layer1': explanation.get('relation_weights_l1', torch.zeros(3)).tolist(),
                'layer2': explanation.get('relation_weights_l2', torch.zeros(3)).tolist(),
            }
        }

    def _feature_attribution(self, node_idx, node_features, adj, adj_per_type):
        """Gradient * input attribution for the target node's fraud logit."""
        features_grad = node_features.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        logits, _, _ = self.model(adj, features_grad, training=False, adj_per_type=adj_per_type)
        fraud_logit = logits[node_idx, 1]
        fraud_logit.backward()

        grad = features_grad.grad[node_idx]
        importance = (grad * features_grad[node_idx].detach()).abs()

        topk_vals, topk_idx = torch.topk(importance, min(self.top_k_features, len(importance)))

        attributions = []
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            name = (self.feature_names[idx] if self.feature_names and idx < len(self.feature_names)
                    else f"feature_{idx}")
            attributions.append({
                'feature': name,
                'importance': val,
                'value': float(node_features[node_idx, idx].item()),
                'gradient': float(grad[idx].item()),
            })
        return attributions

    def _edge_attribution(self, node_idx, adj_per_type):
        """Identify top neighbors per relation type using sparse adj."""
        edge_types_names = ['UPU (same product)', 'USU (same star)', 'UVU (same vote)']
        edge_info = []

        for i, (etype, adj_t) in enumerate(adj_per_type.items()):
            indices = adj_t.coalesce().indices()
            # Find neighbors of node_idx in this relation
            mask = indices[0] == node_idx
            neighbors = indices[1, mask].cpu().tolist()

            etype_name = edge_types_names[i] if i < len(edge_types_names) else f"type_{i}"
            for nb in neighbors[:self.top_k_edges // 3]:
                edge_info.append({
                    'neighbor_idx': int(nb),
                    'relation_type': etype_name,
                    'relation_id': i,
                })

        return edge_info[:self.top_k_edges]

    def _subgraph_attribution(self, node_idx, adj_per_type, probs, max_hops=2):
        """BFS to find high-risk subgraph around the target node."""
        visited = {int(node_idx)}
        frontier = [int(node_idx)]
        subgraph_nodes = [{'node': int(node_idx), 'fraud_prob': float(probs[node_idx, 1].item()), 'hop': 0}]

        for hop in range(1, max_hops + 1):
            next_frontier = []
            for n in frontier:
                for etype, adj_t in adj_per_type.items():
                    indices = adj_t.coalesce().indices()
                    mask = indices[0] == n
                    neighbors = indices[1, mask].cpu().tolist()
                    for nb in neighbors:
                        if nb not in visited:
                            visited.add(nb)
                            prob = float(probs[nb, 1].item())
                            if prob > 0.3:  # only include suspicious neighbors
                                subgraph_nodes.append({'node': nb, 'fraud_prob': prob, 'hop': hop})
                                next_frontier.append(nb)
                            if len(visited) > 50:
                                break
                    if len(visited) > 50:
                        break
                if len(visited) > 50:
                    break
            frontier = next_frontier

        subgraph_nodes.sort(key=lambda x: x['fraud_prob'], reverse=True)
        return {
            'center_node': int(node_idx),
            'high_risk_neighbors': subgraph_nodes[:20],
            'total_explored': len(visited),
        }

    def explain_batch(self, node_indices, node_features, adj, adj_per_type, labels):
        """Explain multiple nodes."""
        results = []
        for idx in node_indices:
            results.append(self.explain_node(idx, node_features, adj, adj_per_type, labels))
        return results

    def to_json(self, explanations, filepath):
        """Save explanations to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(explanations, f, indent=2, ensure_ascii=False)
