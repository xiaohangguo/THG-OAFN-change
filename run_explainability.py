"""Generate explainability case studies for TP-THGN v3."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import json
from models.tp_thgn_gpu import create_tp_thgn_gpu_model
from models.tri_explainer_v3 import TriExplainerV3
from utils.data_loader import load_real_dataset, split_data
from utils.graph_utils import build_sparse_adj
from utils.metrics import calculate_metrics

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_real_dataset('Amazon', data_dir='./data')
adj, adj_per_type = build_sparse_adj(
    dataset.edge_index, dataset.num_nodes, edge_types=dataset.edge_types, normalize=True)
adj = adj.to(device)
adj_per_type = {k: v.to(device) for k, v in adj_per_type.items()}
features = torch.FloatTensor(dataset.node_features).to(device)
labels = torch.LongTensor(dataset.labels).to(device)
train_mask, val_mask, test_mask = split_data(dataset)
train_mask_t = torch.BoolTensor(train_mask).to(device)

# Train model
config = {'input_dim': 25, 'hidden_dim': 128, 'output_dim': 2, 'num_relations': 3,
          'dropout': 0.3, 'oversample_ratio': 1.5, 'beta_laplacian': 0.01, 'focal_gamma': 2.0}
model = create_tp_thgn_gpu_model(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

best_val_f1, best_state = 0.0, None
val_mask_t = torch.BoolTensor(val_mask).to(device)
test_mask_t = torch.BoolTensor(test_mask).to(device)

print("Training model for explainability demo...")
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    _, _, losses = model(adj, features, labels, None, adj_per_type=adj_per_type,
                         training=True, train_mask=train_mask_t)
    losses['total_loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    scheduler.step()
    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            preds, probs = model.predict(adj, features, None, adj_per_type)
        from utils.metrics import calculate_metrics
        vm = calculate_metrics(labels[val_mask_t].cpu().numpy(),
                               preds[val_mask_t].cpu().numpy(),
                               probs[val_mask_t, 1].cpu().numpy())
        if vm['F1'] > best_val_f1:
            best_val_f1 = vm['F1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if best_state:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
print(f"Best val F1: {best_val_f1:.4f}")

# Get predictions on test set
model.eval()
with torch.no_grad():
    preds, probs = model.predict(adj, features, None, adj_per_type)

# Find interesting cases in test set
test_indices = torch.where(test_mask_t)[0]
test_fraud = test_indices[(labels[test_indices] == 1)]
test_normal = test_indices[(labels[test_indices] == 0)]

# Case 1: True positive (correctly identified fraud)
tp_mask = (preds[test_fraud] == 1)
tp_nodes = test_fraud[tp_mask]
# Pick node with highest fraud probability
tp_probs = probs[tp_nodes, 1]
case1_node = tp_nodes[tp_probs.argmax()].item()

# Case 2: False negative (missed fraud)
fn_mask = (preds[test_fraud] == 0)
fn_nodes = test_fraud[fn_mask]
case2_node = fn_nodes[0].item() if len(fn_nodes) > 0 else test_fraud[0].item()

# Case 3: True negative with high confidence
tn_mask = (preds[test_normal] == 0)
tn_nodes = test_normal[tn_mask]
tn_probs = probs[tn_nodes, 0]
case3_node = tn_nodes[tn_probs.argmax()].item()

print(f"\nCase study nodes: TP={case1_node}, FN={case2_node}, TN={case3_node}")

# Generate explanations
explainer = TriExplainerV3(
    model,
    feature_names=[f"feat_{i}" for i in range(25)],
    top_k_features=5,
    top_k_edges=10
)

cases = [case1_node, case2_node, case3_node]
case_names = ['True Positive (Detected Fraud)', 'False Negative (Missed Fraud)', 'True Negative (Normal)']
explanations = []

for node, name in zip(cases, case_names):
    print(f"\n{'='*50}")
    print(f"Case: {name} (node {node})")
    print(f"{'='*50}")
    exp = explainer.explain_node(node, features, adj, adj_per_type, labels)
    exp['case_type'] = name
    explanations.append(exp)
    print(f"  Predicted: {'Fraud' if exp['predicted_label']==1 else 'Normal'} "
          f"(prob={exp['fraud_probability']:.4f})")
    print(f"  True label: {'Fraud' if exp['true_label']==1 else 'Normal'}")
    print(f"  Top features:")
    for fa in exp['feature_attribution'][:3]:
        print(f"    {fa['feature']}: importance={fa['importance']:.4f}, value={fa['value']:.3f}")
    print(f"  Relation weights L1: {[f'{w:.3f}' for w in exp['relation_weights']['layer1']]}")
    print(f"  High-risk neighbors: {len(exp['subgraph_attribution']['high_risk_neighbors'])}")

# Save
os.makedirs('experiments/results', exist_ok=True)
explainer.to_json(explanations, 'experiments/results/explainability_cases.json')
print(f"\nSaved to experiments/results/explainability_cases.json")
