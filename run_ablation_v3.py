"""
TP-THGN v3 Ablation Study.
Tests contribution of each component by removing it one at a time.
Variants:
  1. Full model (all components)
  2. No Graph (remove GatedGraphLayers → pure MLP)
  3. No TP-GraphSMOTE (oversample_ratio=1.0)
  4. No Focal Loss (use standard CE)
  5. No Gating (replace gate with fixed 0.5 weight)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.tp_thgn_gpu import (
    TP_THGN_GPU, FocalLoss, GatedGraphLayer, TimeDegradation,
    create_tp_thgn_gpu_model
)
from models.tp_graphsmote_gpu import TPGraphSMOTEGPU
from utils.data_loader import load_real_dataset, split_data
from utils.graph_utils import build_sparse_adj
from utils.metrics import calculate_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

dataset = load_real_dataset('Amazon', data_dir='./data')
adj, adj_per_type = build_sparse_adj(
    dataset.edge_index, dataset.num_nodes,
    edge_types=dataset.edge_types, normalize=True
)
adj = adj.to(device)
adj_per_type = {k: v.to(device) for k, v in adj_per_type.items()}
features = torch.FloatTensor(dataset.node_features).to(device)
labels = torch.LongTensor(dataset.labels).to(device)

train_mask, val_mask, test_mask = split_data(dataset)
train_mask_t = torch.BoolTensor(train_mask).to(device)
val_mask_t = torch.BoolTensor(val_mask).to(device)
test_mask_t = torch.BoolTensor(test_mask).to(device)


class NoGraphModel(nn.Module):
    """Ablation: remove graph layers entirely (pure MLP)."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3,
                 oversample_ratio=1.5, beta_laplacian=0.01, focal_gamma=2.0):
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
        self.tp_graphsmote = TPGraphSMOTEGPU(hidden_dim, 5, beta_laplacian)
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
        logits = self.classifier(h)
        losses = {}
        if training and labels is not None:
            train_logits = logits[train_mask] if train_mask is not None else logits
            train_labels = labels[train_mask] if train_mask is not None else labels
            lap_loss = torch.tensor(0.0, device=h.device)
            if self.oversample_ratio > 1.0:
                num_fraud = (train_labels == 1).sum().item()
                if num_fraud > 0:
                    train_h = h[train_mask] if train_mask is not None else h
                    h_os, labels_os, _, lap_loss = self.tp_graphsmote(
                        train_h, train_labels, adj, self.oversample_ratio)
                    os_logits = self.classifier(h_os[len(train_labels):])
                    os_labels = labels_os[len(train_labels):]
                    if os_logits.shape[0] > 0:
                        train_logits = torch.cat([train_logits, os_logits], dim=0)
                        train_labels = torch.cat([train_labels, os_labels], dim=0)
            n_f = (train_labels == 1).sum().float()
            n_n = (train_labels == 0).sum().float()
            cw = torch.tensor([1.0, torch.clamp((n_n/n_f).sqrt(), 1.5, 5.0).item()], device=logits.device) if n_f > 0 else None
            loss_cls = FocalLoss(alpha=cw, gamma=self.focal_gamma)(train_logits, train_labels)
            losses['total_loss'] = loss_cls + self.beta_laplacian * lap_loss
        return logits, h, losses

    def predict(self, adj, features, timestamps=None, adj_per_type=None):
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(adj, features, training=False)
            probs = F.softmax(logits, dim=1)
            return probs.argmax(1), probs


class NoGateModel(TP_THGN_GPU):
    """Ablation: fixed gate=0.5 instead of learnable."""
    def forward(self, adj, node_features, labels=None, timestamps=None,
                adj_per_type=None, training=True, train_mask=None):
        h = self.feature_encoder(node_features)
        h = self.time_decay(h, timestamps)
        if adj_per_type is not None:
            weights = F.softmax(self.graph_layer_1.relation_weights, dim=0)
            h_neigh = torch.zeros_like(h)
            for i, (etype, adj_t) in enumerate(adj_per_type.items()):
                if i < len(weights):
                    h_neigh = h_neigh + weights[i] * torch.sparse.mm(adj_t, h)
            h_neigh = self.graph_layer_1.neigh_transform(h_neigh)
            h = h + 0.5 * h_neigh  # fixed gate instead of learnable

            weights2 = F.softmax(self.graph_layer_2.relation_weights, dim=0)
            h_neigh2 = torch.zeros_like(h)
            for i, (etype, adj_t) in enumerate(adj_per_type.items()):
                if i < len(weights2):
                    h_neigh2 = h_neigh2 + weights2[i] * torch.sparse.mm(adj_t, h)
            h_neigh2 = self.graph_layer_2.neigh_transform(h_neigh2)
            h = h + 0.5 * h_neigh2

        logits = self.classifier(h)
        losses = {}
        if training and labels is not None:
            train_logits = logits[train_mask] if train_mask is not None else logits
            train_labels = labels[train_mask] if train_mask is not None else labels
            lap_loss = torch.tensor(0.0, device=h.device)
            if self.oversample_ratio > 1.0:
                num_fraud = (train_labels == 1).sum().item()
                if num_fraud > 0:
                    train_h = h[train_mask] if train_mask is not None else h
                    h_os, labels_os, _, lap_loss = self.tp_graphsmote(
                        train_h, train_labels, adj, self.oversample_ratio)
                    os_logits = self.classifier(h_os[len(train_labels):])
                    os_labels = labels_os[len(train_labels):]
                    if os_logits.shape[0] > 0:
                        train_logits = torch.cat([train_logits, os_logits], dim=0)
                        train_labels = torch.cat([train_labels, os_labels], dim=0)
            n_f = (train_labels == 1).sum().float()
            n_n = (train_labels == 0).sum().float()
            cw = torch.tensor([1.0, torch.clamp((n_n/n_f).sqrt(), 1.5, 5.0).item()], device=logits.device) if n_f > 0 else None
            focal = FocalLoss(alpha=cw, gamma=self.focal_gamma)
            loss_cls = focal(train_logits, train_labels)
            losses['total_loss'] = loss_cls + self.beta_laplacian * lap_loss
        return logits, h, losses


def train_model(model, name, epochs=300, lr=0.005):
    """Train and evaluate a model variant."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    seeds = [42, 123, 456]
    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

        best_val_f1 = 0.0
        best_state = None
        patience = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits, _, losses = model(
                adj, features, labels, None,
                adj_per_type=adj_per_type, training=True, train_mask=train_mask_t
            )
            loss = losses['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    preds, probs = model.predict(adj, features, None, adj_per_type)
                y_true = labels[val_mask_t].cpu().numpy()
                y_pred = preds[val_mask_t].cpu().numpy()
                y_prob = probs[val_mask_t, 1].cpu().numpy()
                val_m = calculate_metrics(y_true, y_pred, y_prob)
                if val_m['F1'] > best_val_f1:
                    best_val_f1 = val_m['F1']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 25:
                    break

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            preds, probs = model.predict(adj, features, None, adj_per_type)
        y_true = labels[test_mask_t].cpu().numpy()
        y_pred = preds[test_mask_t].cpu().numpy()
        y_prob = probs[test_mask_t, 1].cpu().numpy()
        test_m = calculate_metrics(y_true, y_pred, y_prob)
        all_results.append(test_m)

    metrics_keys = ['F1', 'AUC', 'Recall', 'Precision', 'AUPRC', 'G-Mean']
    summary = {}
    for k in metrics_keys:
        vals = [r[k] for r in all_results]
        summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    print(f"  {name:30s} | F1: {summary['F1']['mean']:.4f}±{summary['F1']['std']:.4f} | "
          f"AUC: {summary['AUC']['mean']:.4f} | Recall: {summary['Recall']['mean']:.3f} | Prec: {summary['Precision']['mean']:.3f}")
    return {'model': name, 'summary': summary, 'per_seed': [{k: v for k, v in r.items() if k != 'Confusion_Matrix'} for r in all_results]}


print("="*80)
print("TP-THGN v3 ABLATION STUDY")
print("="*80)

results = []

# 1. Full model
print("\n[1/5] Full model (TP-THGN v3)...")
config = {'input_dim': 25, 'hidden_dim': 128, 'output_dim': 2, 'num_relations': 3,
          'dropout': 0.3, 'oversample_ratio': 1.5, 'beta_laplacian': 0.01, 'focal_gamma': 2.0}
model_full = create_tp_thgn_gpu_model(config)
results.append(train_model(model_full, "TP-THGN v3 (Full)"))

# 2. No Graph Enhancement
print("\n[2/5] No Graph (pure MLP with SMOTE)...")
model_no_graph = NoGraphModel(25, 128, 2, dropout=0.3, oversample_ratio=1.5)
results.append(train_model(model_no_graph, "w/o Graph Enhancement"))

# 3. No TP-GraphSMOTE
print("\n[3/5] No TP-GraphSMOTE...")
config_no_smote = config.copy()
config_no_smote['oversample_ratio'] = 1.0
model_no_smote = create_tp_thgn_gpu_model(config_no_smote)
results.append(train_model(model_no_smote, "w/o TP-GraphSMOTE"))

# 4. No Focal Loss (CE only)
print("\n[4/5] No Focal Loss (CE with class weights)...")
config_no_focal = config.copy()
config_no_focal['focal_gamma'] = 0.0  # gamma=0 reduces focal to CE
model_no_focal = create_tp_thgn_gpu_model(config_no_focal)
results.append(train_model(model_no_focal, "w/o Focal Loss (CE)"))

# 5. No Gating mechanism
print("\n[5/5] No Gating (fixed 0.5 weight)...")
model_no_gate = NoGateModel(
    input_dim=25, hidden_dim=128, output_dim=2, num_relations=3,
    dropout=0.3, oversample_ratio=1.5, beta_laplacian=0.01, focal_gamma=2.0
)
results.append(train_model(model_no_gate, "w/o Learnable Gate"))

# Summary table
print("\n" + "="*80)
print("ABLATION SUMMARY")
print("="*80)
print(f"{'Variant':35s} | {'F1':12s} | {'AUC':12s} | {'Recall':8s} | {'Prec':8s}")
print("-"*80)
for r in results:
    s = r['summary']
    print(f"  {r['model']:33s} | {s['F1']['mean']:.4f}±{s['F1']['std']:.4f} | "
          f"{s['AUC']['mean']:.4f}±{s['AUC']['std']:.4f} | {s['Recall']['mean']:.4f} | {s['Precision']['mean']:.4f}")

full_f1 = results[0]['summary']['F1']['mean']
print(f"\n  Component contributions (ΔF1 vs Full):")
for r in results[1:]:
    delta = full_f1 - r['summary']['F1']['mean']
    print(f"    {r['model']:33s}: ΔF1 = {delta:+.4f}")

os.makedirs('experiments/results', exist_ok=True)
with open('experiments/results/ablation_v3_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to experiments/results/ablation_v3_results.json")
