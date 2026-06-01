"""
GPU-compatible TP-THGN: Topology-Preserving Temporal Heterogeneous Graph Network.
All operations use PyTorch native ops (sparse mm) for full CUDA support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .td_gru_gnn_gpu import TD_GRU_GNN_GPU
from .tp_graphsmote_gpu import TPGraphSMOTEGPU
from .xattention_gpu import XMultiLayerAttentionGPU
from .tri_explainer import TriExplainer


class TP_THGN_GPU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=4,
                 num_heads=8, num_hops=2, dropout=0.5,
                 oversample_ratio=2.0, beta_laplacian=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.oversample_ratio = oversample_ratio
        self.beta_laplacian = beta_laplacian

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.td_gru_gnn = TD_GRU_GNN_GPU(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2
        )

        self.tp_graphsmote = TPGraphSMOTEGPU(
            embedding_dim=hidden_dim,
            k_neighbors=5,
            beta=beta_laplacian
        )

        self.xattention = XMultiLayerAttentionGPU(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_hops=num_hops,
            num_heads=num_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, node_features, labels=None, timestamps=None,
                adj_per_type=None, training=True):
        """
        Args:
            adj: sparse adjacency matrix (N, N) on device
            node_features: (N, input_dim) on device
            labels: (N,) on device
            timestamps: (N,) on device, or None
            adj_per_type: dict {etype: sparse_adj} on device, or None
            training: bool
        """
        h = self.feature_extractor(node_features)
        h = self.td_gru_gnn(adj, h, timestamps)

        lap_loss = torch.tensor(0.0, device=h.device)
        labels_processed = labels
        graph_changed = False

        if training and labels is not None and self.oversample_ratio > 1.0:
            num_fraud = (labels == 1).sum().item()
            if num_fraud > 0:
                h_os, labels_os, _, lap_loss = self.tp_graphsmote(
                    h, labels, adj, self.oversample_ratio
                )
                if h_os.shape[0] != h.shape[0]:
                    graph_changed = True
                h = h_os
                labels_processed = labels_os

        if not graph_changed:
            h = self.xattention(adj, h, adj_per_type)

        h = self.dropout(h)
        logits = self.classifier(h)

        losses = {}
        if training and labels_processed is not None:
            num_fraud = (labels_processed == 1).sum().float()
            num_normal = (labels_processed == 0).sum().float()
            if num_fraud > 0 and num_normal > 0:
                weight_ratio = (num_normal / num_fraud).sqrt()
                weight_fraud = torch.clamp(weight_ratio, min=1.5, max=5.0).item()
                class_weights = torch.tensor([1.0, weight_fraud], device=logits.device)
            else:
                class_weights = None

            loss_cls = F.cross_entropy(logits, labels_processed, weight=class_weights)
            losses['loss_cls'] = loss_cls
            losses['loss_laplacian'] = lap_loss
            losses['total_loss'] = loss_cls + self.beta_laplacian * lap_loss

        return logits, h, losses

    def predict(self, adj, node_features, timestamps=None, adj_per_type=None):
        self.eval()
        with torch.no_grad():
            logits, emb, _ = self.forward(
                adj, node_features, labels=None, timestamps=timestamps,
                adj_per_type=adj_per_type, training=False
            )
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

    def predict_with_explanation(self, adj, node_features, timestamps=None, adj_per_type=None):
        self.eval()
        with torch.no_grad():
            h = self.feature_extractor(node_features)
            h = self.td_gru_gnn(adj, h, timestamps)
            h, explanation = self.xattention(adj, h, adj_per_type, explain=True)
            h = self.dropout(h)
            logits = self.classifier(h)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs, explanation


def create_tp_thgn_gpu_model(config):
    return TP_THGN_GPU(
        input_dim=config.get('input_dim', 25),
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('output_dim', 2),
        num_relations=config.get('num_relations', 3),
        num_heads=config.get('num_heads', 8),
        num_hops=config.get('num_hops', 2),
        dropout=config.get('dropout', 0.5),
        oversample_ratio=config.get('oversample_ratio', 2.0),
        beta_laplacian=config.get('beta_laplacian', 0.01),
    )
