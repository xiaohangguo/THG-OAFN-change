"""
GPU-compatible TP-THGN v3: Topology-Preserving Temporal Heterogeneous Graph Network.
Feature-dominant architecture with gated graph enhancement.
Designed for low-homophily fraud graphs where node features are highly discriminative.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tp_graphsmote_gpu import TPGraphSMOTEGPU


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class GatedGraphLayer(nn.Module):
    """Graph aggregation with learnable gate controlling neighbor influence."""
    def __init__(self, hidden_dim, num_relations=3):
        super().__init__()
        self.relation_weights = nn.Parameter(torch.ones(num_relations) / num_relations)
        self.neigh_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, h, adj_per_type):
        weights = F.softmax(self.relation_weights, dim=0)
        h_neigh = torch.zeros_like(h)
        for i, (etype, adj_t) in enumerate(adj_per_type.items()):
            if i < len(weights):
                h_neigh = h_neigh + weights[i] * torch.sparse.mm(adj_t, h)
        h_neigh = self.neigh_transform(h_neigh)
        g = self.gate(torch.cat([h, h_neigh], dim=1))
        return h + g * h_neigh, weights.detach()


class TimeDegradation(nn.Module):
    """Time-decay weighting for temporal patterns."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.decay_lambda = nn.Parameter(torch.tensor(0.1))
        self.time_proj = nn.Linear(1, hidden_dim)

    def forward(self, h, timestamps=None):
        if timestamps is None:
            return h
        delta_t = 1.0 - timestamps.unsqueeze(1)
        decay = torch.exp(-F.softplus(self.decay_lambda) * delta_t)
        time_signal = self.time_proj(delta_t)
        return h * decay + 0.1 * time_signal


class TP_THGN_GPU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=3,
                 num_heads=8, num_hops=2, dropout=0.3,
                 oversample_ratio=1.5, beta_laplacian=0.01,
                 focal_gamma=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.oversample_ratio = oversample_ratio
        self.beta_laplacian = beta_laplacian
        self.focal_gamma = focal_gamma

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.time_decay = TimeDegradation(hidden_dim)
        self.graph_layer_1 = GatedGraphLayer(hidden_dim, num_relations)
        self.graph_layer_2 = GatedGraphLayer(hidden_dim, num_relations)

        self.tp_graphsmote = TPGraphSMOTEGPU(
            embedding_dim=hidden_dim,
            k_neighbors=5,
            beta=beta_laplacian
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, adj, node_features, labels=None, timestamps=None,
                adj_per_type=None, training=True, train_mask=None):
        h = self.feature_encoder(node_features)
        h = self.time_decay(h, timestamps)

        if adj_per_type is not None:
            h, rel_w1 = self.graph_layer_1(h, adj_per_type)
            h, rel_w2 = self.graph_layer_2(h, adj_per_type)

        logits = self.classifier(h)

        losses = {}
        if training and labels is not None:
            if train_mask is not None:
                train_logits = logits[train_mask]
                train_labels = labels[train_mask]
            else:
                train_logits = logits
                train_labels = labels

            lap_loss = torch.tensor(0.0, device=h.device)
            if self.oversample_ratio > 1.0:
                num_fraud = (train_labels == 1).sum().item()
                if num_fraud > 0:
                    train_h = h[train_mask] if train_mask is not None else h
                    h_os, labels_os, _, lap_loss = self.tp_graphsmote(
                        train_h, train_labels, adj, self.oversample_ratio
                    )
                    os_logits = self.classifier(h_os[len(train_labels):])
                    os_labels = labels_os[len(train_labels):]
                    if os_logits.shape[0] > 0:
                        train_logits = torch.cat([train_logits, os_logits], dim=0)
                        train_labels = torch.cat([train_labels, os_labels], dim=0)

            num_fraud_l = (train_labels == 1).sum().float()
            num_normal_l = (train_labels == 0).sum().float()
            if num_fraud_l > 0 and num_normal_l > 0:
                w = torch.clamp((num_normal_l / num_fraud_l).sqrt(), 1.5, 5.0).item()
                class_weights = torch.tensor([1.0, w], device=logits.device)
            else:
                class_weights = None

            focal = FocalLoss(alpha=class_weights, gamma=self.focal_gamma)
            loss_cls = focal(train_logits, train_labels)
            losses['loss_cls'] = loss_cls
            losses['loss_laplacian'] = lap_loss
            losses['total_loss'] = loss_cls + self.beta_laplacian * lap_loss

        return logits, h, losses

    def predict(self, adj, node_features, timestamps=None, adj_per_type=None):
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(
                adj, node_features, labels=None, timestamps=timestamps,
                adj_per_type=adj_per_type, training=False
            )
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

    def predict_with_explanation(self, adj, node_features, timestamps=None, adj_per_type=None):
        self.eval()
        with torch.no_grad():
            h = self.feature_encoder(node_features)
            h = self.time_decay(h, timestamps)
            explanation = {}
            if adj_per_type is not None:
                h, rel_w1 = self.graph_layer_1(h, adj_per_type)
                h, rel_w2 = self.graph_layer_2(h, adj_per_type)
                explanation['relation_weights_l1'] = rel_w1
                explanation['relation_weights_l2'] = rel_w2
            logits = self.classifier(h)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs, explanation


def create_tp_thgn_gpu_model(config):
    return TP_THGN_GPU(
        input_dim=config.get('input_dim', 25),
        hidden_dim=config.get('hidden_dim', 128),
        output_dim=config.get('output_dim', 2),
        num_relations=config.get('num_relations', 3),
        num_heads=config.get('num_heads', 8),
        num_hops=config.get('num_hops', 2),
        dropout=config.get('dropout', 0.3),
        oversample_ratio=config.get('oversample_ratio', 1.5),
        beta_laplacian=config.get('beta_laplacian', 0.01),
        focal_gamma=config.get('focal_gamma', 2.0),
    )
