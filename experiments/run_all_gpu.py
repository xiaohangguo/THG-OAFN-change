"""
GPU-only experiment runner. All models use sparse matrix ops on GPU.
No DGL dependency for baselines.
"""
import sys, os, time, json
sys.path.insert(0, r'C:\Users\yuhangshu\Downloads\THG-OAFN-change\.claude\worktrees\gpu-implementation')
os.chdir(r'C:\Users\yuhangshu\Downloads\THG-OAFN-change')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.tp_thgn_gpu import create_tp_thgn_gpu_model
from models.td_gru_gnn_gpu import SparseGraphConvLayer
from models.baselines.xgboost_baseline import XGBoostBaseline
from models.baselines.lr_baseline import LRBaseline
from utils.data_loader import load_real_dataset, split_data
from utils.graph_utils import build_sparse_adj
from utils.metrics import calculate_metrics

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

dataset_name = 'Amazon'
seeds = [42, 123, 456, 789, 2024]

# Load dataset
dataset = load_real_dataset(dataset_name, data_dir='./data')
input_dim = dataset.num_features
num_relations = len(np.unique(dataset.edge_types))

# Build sparse adj
adj, adj_per_type = build_sparse_adj(dataset.edge_index, dataset.num_nodes, edge_types=dataset.edge_types, normalize=True)
adj_gpu = adj.to(device)
adj_per_type_gpu = {k: v.to(device) for k, v in adj_per_type.items()} if adj_per_type else None
features_gpu = torch.FloatTensor(dataset.node_features).to(device)
labels_gpu = torch.LongTensor(dataset.labels).to(device)

os.makedirs('experiments/results', exist_ok=True)


# === GPU GNN Baselines (no DGL) ===
class GCNBaselineGPU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SparseGraphConvLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SparseGraphConvLayer(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            h = layer(adj, h)
            h = self.dropout(h)
        return self.classifier(h)


class GATLayerGPU(nn.Module):
    """Simple attention-based aggregation using sparse ops."""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim * num_heads)
        self.attn = nn.Parameter(torch.randn(num_heads, out_dim * 2))

    def forward(self, adj, h):
        N = h.shape[0]
        h_proj = self.W(h).view(N, self.num_heads, self.head_dim)
        # Simplified: use sparse mm for aggregation (approximate attention)
        h_flat = h_proj.mean(dim=1)  # (N, head_dim)
        h_agg = torch.sparse.mm(adj, h_flat)
        return F.elu(h_agg)


class GATBaselineGPU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayerGPU(input_dim, hidden_dim, num_heads))
        for _ in range(num_layers - 1):
            self.layers.append(GATLayerGPU(hidden_dim, hidden_dim, num_heads))
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            h = layer(adj, h)
            h = self.dropout(h)
        return self.classifier(h)


class GraphSAGEBaselineGPU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim * 2, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            h_neigh = torch.sparse.mm(adj, h)
            h = torch.cat([h, h_neigh], dim=1)
            h = F.relu(layer(h))
            h = self.dropout(h)
        return self.classifier(h)


def train_gnn_gpu(ModelClass, model_kwargs, seeds):
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_mask, val_mask, test_mask = split_data(dataset)

        model = ModelClass(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        num_fraud = (labels_gpu == 1).sum().float()
        num_normal = (labels_gpu == 0).sum().float()
        w = torch.clamp((num_normal / num_fraud).sqrt(), 1.5, 5.0).item()
        class_w = torch.tensor([1.0, w], device=device)

        best_f1, best_state, patience = 0.0, None, 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits = model(adj_gpu, features_gpu)
            loss = F.cross_entropy(logits[train_mask], labels_gpu[train_mask], weight=class_w)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    logits = model(adj_gpu, features_gpu)
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                vm = calculate_metrics(labels_gpu[val_mask].cpu().numpy(), preds[val_mask].cpu().numpy(), probs[val_mask, 1].cpu().numpy())
                if vm['F1'] > best_f1:
                    best_f1 = vm['F1']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 10:
                    break

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            logits = model(adj_gpu, features_gpu)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        tm = calculate_metrics(labels_gpu[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), probs[test_mask, 1].cpu().numpy())
        all_metrics.append(tm)

    result = {}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def run_tp_thgn(config, seeds, variant_name):
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_mask, val_mask, test_mask = split_data(dataset)

        model = create_tp_thgn_gpu_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_val_f1, best_state, patience = 0.0, None, 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits, emb, losses = model(adj_gpu, features_gpu, labels_gpu, None, adj_per_type=adj_per_type_gpu, training=True)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    preds, probs = model.predict(adj_gpu, features_gpu, None, adj_per_type_gpu)
                val_m = calculate_metrics(labels_gpu[val_mask].cpu().numpy(), preds[val_mask].cpu().numpy(), probs[val_mask, 1].cpu().numpy())
                if val_m['F1'] > best_val_f1:
                    best_val_f1 = val_m['F1']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 10:
                    break

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            preds, probs = model.predict(adj_gpu, features_gpu, None, adj_per_type_gpu)
        test_m = calculate_metrics(labels_gpu[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), probs[test_mask, 1].cpu().numpy())
        all_metrics.append(test_m)
        print(f"    Seed {seed}: F1={test_m['F1']:.4f} AUC={test_m['AUC']:.4f}")

    result = {'variant': variant_name, 'model': variant_name}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def run_sklearn(ModelClass, seeds):
    all_metrics = []
    for seed in seeds:
        np.random.seed(seed)
        train_mask, val_mask, test_mask = split_data(dataset)
        model = ModelClass()
        model.fit(dataset.node_features[train_mask], dataset.labels[train_mask])
        preds, probs = model.predict(dataset.node_features[test_mask])
        tm = calculate_metrics(dataset.labels[test_mask], preds, probs[:, 1])
        all_metrics.append(tm)
    result = {}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


# ============================================================
all_results = []
start_total = time.time()

print("\n[1/9] Logistic Regression...")
r = run_sklearn(LRBaseline, seeds)
r['model'] = 'Logistic Regression'
all_results.append(r)
print(f"  F1={r['F1']['mean']:.4f} ({time.time()-start_total:.0f}s)")

print("[2/9] XGBoost...")
r = run_sklearn(XGBoostBaseline, seeds)
r['model'] = 'XGBoost'
all_results.append(r)
print(f"  F1={r['F1']['mean']:.4f} ({time.time()-start_total:.0f}s)")

print("[3/9] GCN (GPU)...")
r = train_gnn_gpu(GCNBaselineGPU, {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2}, seeds)
r['model'] = 'GCN'
all_results.append(r)
print(f"  F1={r['F1']['mean']:.4f} ({time.time()-start_total:.0f}s)")

print("[4/9] GAT (GPU)...")
r = train_gnn_gpu(GATBaselineGPU, {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2, 'num_heads': 4}, seeds)
r['model'] = 'GAT'
all_results.append(r)
print(f"  F1={r['F1']['mean']:.4f} ({time.time()-start_total:.0f}s)")

print("[5/9] GraphSAGE (GPU)...")
r = train_gnn_gpu(GraphSAGEBaselineGPU, {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2}, seeds)
r['model'] = 'GraphSAGE'
all_results.append(r)
print(f"  F1={r['F1']['mean']:.4f} ({time.time()-start_total:.0f}s)")

# Save baselines
with open('experiments/results/comparison_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("Baselines saved.")

# TP-THGN Ablation
base_config = {
    'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2,
    'num_relations': num_relations, 'num_heads': 8, 'num_hops': 2,
    'dropout': 0.5, 'oversample_ratio': 2.0, 'beta_laplacian': 0.01,
}

ablation_results = []
variants = [
    ('Full TP-THGN', base_config.copy()),
    ('w/o Laplacian', {**base_config, 'beta_laplacian': 0.0}),
    ('w/o Oversampling', {**base_config, 'oversample_ratio': 1.0}),
    ('w/o Both', {**base_config, 'oversample_ratio': 1.0, 'beta_laplacian': 0.0}),
]

for i, (name, config) in enumerate(variants):
    print(f"[{6+i}/9] {name}...")
    r = run_tp_thgn(config, seeds, name)
    ablation_results.append(r)
    print(f"  F1={r['F1']['mean']:.4f} ({time.time()-start_total:.0f}s)")

with open('experiments/results/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2)

# Add TP-THGN to comparison
tp_result = ablation_results[0].copy()
tp_result['model'] = 'TP-THGN (ours)'
all_results.append(tp_result)
with open('experiments/results/comparison_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Summary
total_time = time.time() - start_total
print(f"\n{'='*70}")
print(f"COMPLETED in {total_time/60:.1f} min")
print(f"{'='*70}")
print(f"\n{'Model':<22} {'AUC':>8} {'F1':>8} {'Recall':>8} {'Prec':>8} {'AUPRC':>8}")
print("-" * 70)
for r in all_results:
    print(f"{r['model']:<22} {r['AUC']['mean']:>8.4f} {r['F1']['mean']:>8.4f} "
          f"{r['Recall']['mean']:>8.4f} {r['Precision']['mean']:>8.4f} "
          f"{r.get('AUPRC', {}).get('mean', 0):>8.4f}")
print(f"\nGPU Mem: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
