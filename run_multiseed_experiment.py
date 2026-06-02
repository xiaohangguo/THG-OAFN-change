"""Multi-seed comparison experiment: TP-THGN v3 vs baselines."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import numpy as np
from models.tp_thgn_gpu import create_tp_thgn_gpu_model
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

config = {
    'input_dim': 25, 'hidden_dim': 128, 'output_dim': 2,
    'num_relations': 3, 'num_heads': 8, 'num_hops': 2,
    'dropout': 0.3, 'oversample_ratio': 1.5,
    'beta_laplacian': 0.01, 'focal_gamma': 2.0,
}

seeds = [42, 123, 456]
all_results = []

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = create_tp_thgn_gpu_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_f1 = 0.0
    best_state = None
    patience = 0

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        logits, emb, losses = model(
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
    print(f"  Seed {seed}: F1={test_m['F1']:.4f}, AUC={test_m['AUC']:.4f}, Recall={test_m['Recall']:.4f}, Prec={test_m['Precision']:.4f}")

print("\n" + "="*60)
print("TP-THGN v3 MULTI-SEED RESULTS")
print("="*60)
metrics_keys = ['F1', 'AUC', 'Recall', 'Precision', 'AUPRC', 'G-Mean']
summary = {}
for k in metrics_keys:
    vals = [r[k] for r in all_results]
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    summary[k] = {'mean': mean_v, 'std': std_v}
    print(f"  {k}: {mean_v:.4f} +/- {std_v:.4f}")

print("\n--- Comparison with baselines ---")
print(f"  TP-THGN v3:   F1={summary['F1']['mean']:.4f} +/- {summary['F1']['std']:.4f}")
print(f"  TP-THGN v1:   F1=0.6500 (old, broken)")
print(f"  GraphSAGE:    F1=0.9130")
print(f"  XGBoost:      F1=0.9213")
print(f"  GCN:          F1=0.4453")
print(f"  GAT:          F1=0.4030")

improvement_vs_gcn = (summary['F1']['mean'] - 0.4453) * 100
improvement_vs_thgoafn = (summary['F1']['mean'] - 0.6500) * 100  # proxy
print(f"\n  Improvement vs GCN: +{improvement_vs_gcn:.1f}pp")
print(f"  Improvement vs v1 (proxy THG-OAFN): +{improvement_vs_thgoafn:.1f}pp")

os.makedirs('experiments/results', exist_ok=True)
final_result = {
    'model': 'TP-THGN-v3',
    'dataset': 'Amazon',
    'config': config,
    'seeds': seeds,
    'per_seed_results': [{k: v for k, v in r.items() if k != 'Confusion_Matrix'} for r in all_results],
    'summary': {k: {'mean': float(v['mean']), 'std': float(v['std'])} for k, v in summary.items()},
}
with open('experiments/results/tp_thgn_v3_multiseed.json', 'w') as f:
    json.dump(final_result, f, indent=2)
print("\nResults saved to experiments/results/tp_thgn_v3_multiseed.json")
