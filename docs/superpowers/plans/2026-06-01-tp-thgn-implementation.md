# TP-THGN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the TP-THGN model with four improvement modules, run ablation/comparison experiments, and produce all paper figures/tables.

**Architecture:** Modular enhancement of existing THG-OAFN — each phase adds one module (TD-GRU-GNN, TP-GraphSMOTE, XAttention, TriExplainer) on a separate git branch, with final phases running ablation/comparison experiments and cross-dataset validation.

**Tech Stack:** Python 3.10, PyTorch 2.5+cu124, DGL 2.0, scikit-learn, matplotlib, seaborn. Conda env: `thg-oafn`. GPU: RTX 4070 Laptop.

---

## File Structure

```
THG-OAFN-change/
├── models/
│   ├── thg_oafn.py              # Original (keep unchanged as baseline B6)
│   ├── gru_gnn.py               # Original GRU-GNN (keep)
│   ├── graph_smote.py           # Original GraphSMOTE (keep)
│   ├── attention.py             # Original attention (keep)
│   ├── td_gru_gnn.py           # NEW: Time-Decay GRU-GNN
│   ├── tp_graphsmote.py         # NEW: Topology-Preserving GraphSMOTE
│   ├── xattention.py           # NEW: Explainable Multi-Layer Attention
│   ├── tri_explainer.py         # NEW: Three-Level Explainer
│   ├── tp_thgn.py              # NEW: Full TP-THGN model
│   └── baselines/
│       ├── __init__.py
│       ├── lr_baseline.py       # Logistic Regression
│       ├── xgboost_baseline.py  # XGBoost
│       ├── gcn_baseline.py      # GCN
│       ├── gat_baseline.py      # GAT
│       └── graphsage_baseline.py # GraphSAGE
├── utils/
│   ├── data_loader.py           # MODIFY: add IEEE-CIS + Kaggle CC loaders
│   ├── metrics.py               # MODIFY: add AUPRC, G-Mean
│   └── visualization.py        # NEW: paper figure generation
├── experiments/
│   ├── run_baseline.py          # NEW: run all baselines
│   ├── run_ablation.py          # NEW: ablation study
│   ├── run_comparison.py        # NEW: full comparison table
│   ├── run_cross_dataset.py     # NEW: cross-dataset validation
│   └── results/                 # Experiment result JSONs (git-tracked)
├── figures/                     # Generated paper figures (git-tracked)
├── train_tp_thgn.py            # NEW: training script for TP-THGN
├── .gitignore                   # MODIFY: add data exclusions
└── requirements.txt             # MODIFY: add xgboost, networkx
```

---

## Phase 0: Baseline & Infrastructure

### Task 0.1: Update .gitignore and requirements

**Files:**
- Modify: `.gitignore`
- Modify: `requirements.txt`

- [ ] **Step 1: Update .gitignore**

```gitignore
# Data files (large)
data/ieee-cis/
data/credit-card/
data/*.csv
*.pth
!checkpoints/check.txt

# Experiment intermediates
experiments/results/*.pth
__pycache__/
*.pyc
```

- [ ] **Step 2: Update requirements.txt**

Append to existing file:
```
xgboost>=1.7.0
networkx>=2.8.0
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore requirements.txt
git commit -m "chore: update gitignore for data files, add xgboost/networkx deps"
```

### Task 0.2: Extend metrics module

**Files:**
- Modify: `utils/metrics.py`

- [ ] **Step 1: Add AUPRC and G-Mean to calculate_metrics**

Add after the existing F1 calculation in `calculate_metrics`:

```python
from sklearn.metrics import average_precision_score

# Inside calculate_metrics, after F1 computation:
if y_prob is not None and not (np.isnan(y_prob).any() or np.isinf(y_prob).any()):
    unique_labels = np.unique(y_true)
    if len(unique_labels) >= 2:
        try:
            metrics['AUPRC'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['AUPRC'] = 0.0
    else:
        metrics['AUPRC'] = 0.0
else:
    metrics['AUPRC'] = 0.0

# G-Mean
recall_pos = metrics['Recall']
tn = cm[0, 0] if cm.shape[0] > 1 else 0
fp = cm[0, 1] if cm.shape[0] > 1 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
metrics['G-Mean'] = np.sqrt(recall_pos * specificity)
metrics['Specificity'] = specificity
```

- [ ] **Step 2: Verify metrics still work with existing train.py**

```bash
conda activate thg-oafn
python train.py --epochs 5
```

Expected: Training runs without error, metrics dict now includes AUPRC and G-Mean.

- [ ] **Step 3: Commit**

```bash
git add utils/metrics.py
git commit -m "feat: add AUPRC and G-Mean to evaluation metrics"
```

### Task 0.3: Create experiment infrastructure

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/results/.gitkeep`
- Create: `figures/.gitkeep`

- [ ] **Step 1: Create directories and placeholder files**

```bash
mkdir -p experiments/results figures
touch experiments/__init__.py experiments/results/.gitkeep figures/.gitkeep
```

- [ ] **Step 2: Commit**

```bash
git add experiments/ figures/
git commit -m "chore: add experiment and figures directories"
```

### Task 0.4: Run original THG-OAFN baseline

**Files:**
- Create: `experiments/run_baseline.py`

- [ ] **Step 1: Create baseline runner script**

```python
"""Run baseline models and save results."""
import json
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.thg_oafn import create_thg_oafn_model
from utils.data_loader import load_real_dataset, split_data
from utils.metrics import calculate_metrics
import torch.optim as optim


def run_thg_oafn_baseline(dataset_name='Amazon', seeds=[42, 123, 456, 789, 2024]):
    """Run THG-OAFN with multiple seeds and report mean/std."""
    all_metrics = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = load_real_dataset(dataset_name, data_dir='./data')
        g = dataset.build_dgl_graph()
        features = torch.FloatTensor(dataset.node_features)
        labels = torch.LongTensor(dataset.labels)
        train_mask, val_mask, test_mask = split_data(dataset)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = {
            'input_dim': dataset.num_features,
            'hidden_dim': 64,
            'output_dim': 2,
            'num_relations': len(np.unique(dataset.edge_types)),
            'num_heads': 8,
            'num_hops': 2,
            'dropout': 0.5,
            'oversample_ratio': 1.0,
        }

        model = create_thg_oafn_model(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        features = features.to(device)
        labels = labels.to(device)
        g = g.to(device)

        best_val_f1 = 0.0
        patience = 0

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits, embeddings, losses = model(g, features, labels, training=True)
            loss = losses['total_loss']
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    preds, probs = model.predict(g, features)
                y_true = labels[val_mask].cpu().numpy()
                y_pred = preds[val_mask].cpu().numpy()
                y_prob = probs[val_mask, 1].cpu().numpy()
                val_m = calculate_metrics(y_true, y_pred, y_prob)

                if val_m['F1'] > best_val_f1:
                    best_val_f1 = val_m['F1']
                    best_state = model.state_dict().copy()
                    patience = 0
                else:
                    patience += 1
                if patience >= 10:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds, probs = model.predict(g, features)
        y_true = labels[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
        y_prob = probs[test_mask, 1].cpu().numpy()
        test_m = calculate_metrics(y_true, y_pred, y_prob)
        all_metrics.append(test_m)
        print(f"Seed {seed}: F1={test_m['F1']:.4f} AUC={test_m['AUC']:.4f} AUPRC={test_m.get('AUPRC',0):.4f}")

    # Aggregate
    result = {}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        values = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    result['model'] = 'THG-OAFN (original)'
    result['dataset'] = dataset_name
    result['seeds'] = seeds

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/baseline_thg_oafn.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nTHG-OAFN Baseline Results (mean ± std):")
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        print(f"  {key}: {result[key]['mean']:.4f} ± {result[key]['std']:.4f}")

    return result


if __name__ == '__main__':
    run_thg_oafn_baseline()
```

- [ ] **Step 2: Run baseline**

```bash
conda activate thg-oafn
python experiments/run_baseline.py
```

Expected: Prints metrics for 5 seeds, saves `experiments/results/baseline_thg_oafn.json`.

- [ ] **Step 3: Commit results**

```bash
git add experiments/run_baseline.py experiments/results/baseline_thg_oafn.json
git commit -m "feat: run THG-OAFN baseline, record results (Phase 0)"
```

---

## Phase 1: TD-GRU-GNN (Time-Decay GRU-GNN)

### Task 1.1: Implement TD-GRU-GNN module

**Files:**
- Create: `models/td_gru_gnn.py`

- [ ] **Step 1: Create the time-decay GRU-GNN module**

```python
"""
Time-Decay GRU-GNN Fusion Module (TD-GRU-GNN)
Extends GRU-GNN with learnable temporal decay weighting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=1)
        r_t = torch.sigmoid(self.W_r(combined))
        z_t = torch.sigmoid(self.W_z(combined))
        combined_reset = torch.cat([x_t, r_t * h_prev], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            return F.relu(self.W(h_neigh))


class TD_GRU_GNN(nn.Module):
    """
    Time-Decay GRU-GNN: applies exp(-lambda * delta_t) weighting to
    temporal features before fusing with structural features.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, fusion_alpha=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_alpha = nn.Parameter(torch.tensor(fusion_alpha))
        self.decay_lambda = nn.Parameter(torch.tensor(0.1))

        self.gru = GRULayer(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, g, node_features, timestamps=None):
        """
        Args:
            g: DGL graph
            node_features: (N, input_dim)
            timestamps: (N,) normalized to [0,1], or None

        Returns:
            h_out: (N, output_dim)
        """
        num_nodes = node_features.shape[0]
        device = node_features.device

        h_gru = torch.zeros(num_nodes, self.hidden_dim, device=device)
        h_gru = self.gru(node_features, h_gru)

        # Time decay: recent nodes get higher weight
        if timestamps is not None:
            delta_t = 1.0 - timestamps.unsqueeze(1)  # (N, 1), newer = smaller delta
            decay_weight = torch.exp(-F.softplus(self.decay_lambda) * delta_t)
            h_gru_weighted = h_gru * decay_weight
        else:
            h_gru_weighted = h_gru

        h_gnn = h_gru
        for layer in self.gnn_layers:
            h_gnn = layer(g, h_gnn)

        alpha = torch.sigmoid(self.fusion_alpha)
        h_fused = alpha * h_gru_weighted + (1 - alpha) * h_gnn

        return self.output_proj(h_fused)
```

- [ ] **Step 2: Verify module instantiation and forward pass**

```bash
conda activate thg-oafn
python -c "
import torch, dgl
from models.td_gru_gnn import TD_GRU_GNN
g = dgl.graph(([0,1,2,3],[1,2,3,0]))
x = torch.randn(4, 25)
ts = torch.tensor([0.1, 0.3, 0.7, 0.9])
model = TD_GRU_GNN(25, 64, 64)
out = model(g, x, ts)
assert out.shape == (4, 64), f'Expected (4,64), got {out.shape}'
print('TD-GRU-GNN OK:', out.shape)
"
```

Expected: `TD-GRU-GNN OK: torch.Size([4, 64])`

- [ ] **Step 3: Commit**

```bash
git add models/td_gru_gnn.py
git commit -m "feat: implement TD-GRU-GNN with learnable time decay (Phase 1)"
```

---

## Phase 2: TP-GraphSMOTE (Topology-Preserving GraphSMOTE)

### Task 2.1: Implement TP-GraphSMOTE module

**Files:**
- Create: `models/tp_graphsmote.py`

- [ ] **Step 1: Create topology-preserving GraphSMOTE**

```python
"""
Topology-Preserving GraphSMOTE (TP-GraphSMOTE)
Adds Laplacian regularization to preserve fraud cluster topology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


class TPGraphSMOTE(nn.Module):
    def __init__(self, embedding_dim, k_neighbors=5, beta=0.01):
        """
        Args:
            embedding_dim: Node embedding dimension
            k_neighbors: Number of nearest neighbors for SMOTE
            beta: Laplacian regularization coefficient
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.beta = beta

        self.attribute_completion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        self.edge_decoder_weight = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim) * 0.01
        )

    def forward(self, node_embeddings, labels, adj_matrix, oversample_ratio=2.0):
        """
        Returns:
            new_embeddings, new_labels, new_adj, laplacian_loss
        """
        device = node_embeddings.device
        num_nodes = node_embeddings.shape[0]

        fraud_mask = labels == 1
        fraud_indices = torch.where(fraud_mask)[0]
        fraud_embeddings = node_embeddings[fraud_mask]
        num_fraud = fraud_embeddings.shape[0]

        if num_fraud == 0 or oversample_ratio <= 1.0:
            zero_loss = torch.tensor(0.0, device=device)
            return node_embeddings, labels, adj_matrix, zero_loss

        num_to_generate = int(num_fraud * (oversample_ratio - 1))
        if num_to_generate <= 0:
            zero_loss = torch.tensor(0.0, device=device)
            return node_embeddings, labels, adj_matrix, zero_loss

        # KNN in embedding space
        fraud_np = fraud_embeddings.detach().cpu().numpy()
        k = min(self.k_neighbors, num_fraud)
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(fraud_np)

        new_embeddings_list = []
        parent_indices = []  # track which fraud node spawned each synthetic

        for i in range(num_to_generate):
            idx = np.random.randint(0, num_fraud)
            base = fraud_embeddings[idx:idx + 1]

            _, indices = knn.kneighbors(base.detach().cpu().numpy(), n_neighbors=min(2, num_fraud))
            neighbor_idx = indices[0, 1] if indices.shape[1] > 1 else indices[0, 0]
            neighbor = fraud_embeddings[neighbor_idx:neighbor_idx + 1]

            delta = torch.rand(1, device=device)
            synthetic = (1 - delta) * base + delta * neighbor
            synthetic = self.attribute_completion(synthetic)

            new_embeddings_list.append(synthetic)
            parent_indices.append(fraud_indices[idx].item())

        new_fraud_embs = torch.cat(new_embeddings_list, dim=0)
        new_embeddings = torch.cat([node_embeddings, new_fraud_embs], dim=0)

        new_labels = torch.cat([
            labels,
            torch.ones(num_to_generate, dtype=labels.dtype, device=device)
        ])

        # Edge generation
        new_adj = self._generate_edges(new_embeddings, adj_matrix, num_nodes, num_to_generate)

        # Laplacian regularization: synthetic nodes should be smooth with parents
        lap_loss = self._laplacian_loss(
            new_embeddings, new_adj, num_nodes, num_to_generate, parent_indices
        )

        return new_embeddings, new_labels, new_adj, lap_loss

    def _generate_edges(self, embeddings, old_adj, num_old, num_new):
        device = embeddings.device
        total = num_old + num_new

        new_adj = torch.zeros(total, total, device=device)
        new_adj[:num_old, :num_old] = old_adj

        for i in range(num_old, total):
            emb = embeddings[i:i + 1]
            scores = torch.sigmoid(
                torch.mm(torch.mm(emb, self.edge_decoder_weight), embeddings[:num_old].t())
            ).squeeze()
            edges = (scores > 0.5).float()
            new_adj[i, :num_old] = edges
            new_adj[:num_old, i] = edges

        return new_adj

    def _laplacian_loss(self, embeddings, adj, num_old, num_new, parent_indices):
        """
        L_lap = sum_{synthetic i, neighbor j in adj} ||h_i - h_j||^2 * A_ij
        """
        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for k in range(num_new):
            syn_idx = num_old + k
            syn_emb = embeddings[syn_idx]

            # Get connected nodes from adjacency
            neighbors = torch.where(adj[syn_idx, :num_old] > 0)[0]
            if len(neighbors) == 0:
                # Fallback: use parent node
                parent = parent_indices[k]
                diff = syn_emb - embeddings[parent]
                loss = loss + torch.sum(diff ** 2)
                count += 1
            else:
                for nb in neighbors[:5]:  # cap at 5 neighbors for efficiency
                    diff = syn_emb - embeddings[nb]
                    loss = loss + torch.sum(diff ** 2) * adj[syn_idx, nb]
                    count += 1

        if count > 0:
            loss = loss / count

        return loss
```

- [ ] **Step 2: Verify module**

```bash
conda activate thg-oafn
python -c "
import torch
from models.tp_graphsmote import TPGraphSMOTE
embs = torch.randn(100, 64)
labels = torch.zeros(100, dtype=torch.long)
labels[:7] = 1  # 7% fraud
adj = torch.eye(100)
model = TPGraphSMOTE(64, k_neighbors=3, beta=0.01)
new_embs, new_labels, new_adj, lap_loss = model(embs, labels, adj, oversample_ratio=2.0)
print(f'Original: {embs.shape[0]}, After: {new_embs.shape[0]}')
print(f'Laplacian loss: {lap_loss.item():.4f}')
assert new_embs.shape[0] > embs.shape[0]
assert lap_loss.requires_grad
print('TP-GraphSMOTE OK')
"
```

Expected: Shows increased node count and a non-zero laplacian loss with grad.

- [ ] **Step 3: Commit**

```bash
git add models/tp_graphsmote.py
git commit -m "feat: implement TP-GraphSMOTE with Laplacian regularization (Phase 2)"
```

---

## Phase 3: XAttention (Explainable Multi-Layer Attention)

### Task 3.1: Implement XAttention module

**Files:**
- Create: `models/xattention.py`

- [ ] **Step 1: Create explainable attention module**

```python
"""
Explainable Multi-Layer Attention (XAttention)
Same architecture as original but exposes attention weights for explanation.
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
        edge_attentions = []

        for k in range(self.num_hops):
            agg, attn = self._aggregate(g, current)
            hop_embeddings.append(agg)
            if explain:
                edge_attentions.append(attn)
            current = agg

        hop_w = F.softmax(self.hop_weights, dim=0)
        h_fused = sum(hop_w[k] * hop_embeddings[k] for k in range(self.num_hops))

        if explain:
            return h_fused, {'hop_weights': hop_w.detach(), 'edge_attentions': edge_attentions}
        return h_fused

    def _aggregate(self, g, features):
        head_outputs = []
        attn_scores = None

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

        h_out = torch.mean(torch.stack(head_outputs), dim=0)
        return h_out, attn_scores


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
        h_final = torch.cat([gate_weights[h] * head_outputs[h] for h in range(self.num_heads)], dim=1)

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
        """
        Returns:
            h_final: (N, hidden_dim)
            explanation: dict with attention weights (only if explain=True)
        """
        explanation = {}

        if explain:
            fused_adj, rel_weights = self.relation_fusion(adj_matrices, explain=True)
            explanation['relation_weights'] = rel_weights
        else:
            fused_adj = self.relation_fusion(adj_matrices)

        if explain:
            h_multi, neigh_info = self.neighborhood_fusion(g, node_features, explain=True)
            explanation['neighborhood'] = neigh_info
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
```

- [ ] **Step 2: Verify module with explain mode**

```bash
conda activate thg-oafn
python -c "
import torch, dgl
from models.xattention import XMultiLayerAttention
g = dgl.graph(([0,1,2,3,0,2],[1,2,3,0,3,1]))
x = torch.randn(4, 64)
model = XMultiLayerAttention(64, num_relations=3, num_hops=2, num_heads=8)
# Normal mode
out = model(g, x)
assert out.shape == (4, 64)
# Explain mode
out2, expl = model(g, x, explain=True)
assert 'relation_weights' in expl
assert 'gate_weights' in expl
print('XAttention OK, gate_weights:', expl['gate_weights'])
"
```

- [ ] **Step 3: Commit**

```bash
git add models/xattention.py
git commit -m "feat: implement XAttention with explainable weight export (Phase 3)"
```

---

## Phase 4: TriExplainer (Three-Level Explainer)

### Task 4.1: Implement TriExplainer module

**Files:**
- Create: `models/tri_explainer.py`

- [ ] **Step 1: Create three-level explainer**

```python
"""
Three-Level Explainer (TriExplainer)
Provides feature-level, edge-level, and subgraph-level attribution.
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

    def explain_node(self, node_idx, explanation_dict, g, node_features, predictions, probabilities):
        """
        Generate three-level explanation for a single node.

        Args:
            node_idx: Target node index
            explanation_dict: Output from XAttention(explain=True)
            g: DGL graph
            node_features: (N, F) tensor
            predictions: (N,) predicted labels
            probabilities: (N, 2) predicted probabilities

        Returns:
            dict with feature_attribution, edge_attribution, subgraph_attribution
        """
        result = {
            'node_idx': int(node_idx),
            'predicted_label': int(predictions[node_idx].item()),
            'fraud_probability': float(probabilities[node_idx, 1].item()),
        }

        # Level 1: Feature attribution from gate weights
        result['feature_attribution'] = self._feature_attribution(
            explanation_dict, node_features[node_idx]
        )

        # Level 2: Edge attribution from neighborhood attention
        result['edge_attribution'] = self._edge_attribution(
            node_idx, explanation_dict, g
        )

        # Level 3: Subgraph attribution
        result['subgraph_attribution'] = self._subgraph_attribution(
            node_idx, g, probabilities
        )

        return result

    def _feature_attribution(self, explanation_dict, node_feature):
        """Gate weights indicate which feature dimensions matter most."""
        gate_weights = explanation_dict.get('gate_weights', None)
        if gate_weights is None:
            return []

        # Gate weights are per-head; expand to feature dims
        num_heads = len(gate_weights)
        head_dim = node_feature.shape[0] // num_heads

        # Compute per-dimension importance
        importance = torch.zeros(node_feature.shape[0])
        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            importance[start:end] = gate_weights[h].item() * torch.abs(node_feature[start:end])

        # Top-K
        topk_vals, topk_idx = torch.topk(importance, min(self.top_k_features, len(importance)))

        attributions = []
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            name = self.feature_names[idx] if self.feature_names and idx < len(self.feature_names) else f"dim_{idx}"
            attributions.append({'feature': name, 'importance': val, 'value': float(node_feature[idx])})

        return attributions

    def _edge_attribution(self, node_idx, explanation_dict, g):
        """Identify which neighbor connections contribute most."""
        # Get 1-hop neighbors
        predecessors = g.predecessors(node_idx).cpu().numpy().tolist()
        successors = g.successors(node_idx).cpu().numpy().tolist()
        neighbors = list(set(predecessors + successors))

        if not neighbors:
            return []

        # Relation weights tell us which relation types matter
        rel_weights = explanation_dict.get('relation_weights', None)

        # For each neighbor, compute edge importance based on relation type
        edge_attrs = []
        edge_types = g.edata.get('edge_type', None)

        for nb in neighbors[:self.top_k_edges]:
            attr = {'neighbor_idx': nb}
            # Find edge between node_idx and nb
            src, dst = g.edges()
            mask = ((src == node_idx) & (dst == nb)) | ((src == nb) & (dst == node_idx))
            if mask.any() and edge_types is not None:
                etype = edge_types[mask][0].item()
                attr['edge_type'] = int(etype)
                if rel_weights is not None and etype < len(rel_weights):
                    attr['relation_importance'] = float(rel_weights[etype].item())
            edge_attrs.append(attr)

        # Sort by relation importance if available
        edge_attrs.sort(key=lambda x: x.get('relation_importance', 0), reverse=True)
        return edge_attrs[:self.top_k_edges]

    def _subgraph_attribution(self, node_idx, g, probabilities):
        """Extract minimal subgraph that explains the prediction."""
        # BFS from node_idx, keep nodes with high fraud probability
        visited = {node_idx}
        frontier = [node_idx]
        subgraph_nodes = [node_idx]
        fraud_prob_threshold = self.subgraph_threshold

        for hop in range(2):  # 2-hop subgraph
            next_frontier = []
            for n in frontier:
                neighbors = g.successors(n).cpu().numpy().tolist()
                neighbors += g.predecessors(n).cpu().numpy().tolist()
                for nb in set(neighbors):
                    if nb not in visited:
                        visited.add(nb)
                        if probabilities[nb, 1].item() > fraud_prob_threshold:
                            subgraph_nodes.append(nb)
                            next_frontier.append(nb)
            frontier = next_frontier

        return {
            'center_node': int(node_idx),
            'subgraph_nodes': [int(n) for n in subgraph_nodes],
            'num_nodes': len(subgraph_nodes),
            'max_hops': 2,
        }

    def explain_batch(self, node_indices, explanation_dict, g, node_features, predictions, probabilities):
        """Explain multiple nodes."""
        return [
            self.explain_node(idx, explanation_dict, g, node_features, predictions, probabilities)
            for idx in node_indices
        ]

    def to_json(self, explanations, filepath):
        """Save explanations to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(explanations, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 2: Verify explainer**

```bash
conda activate thg-oafn
python -c "
import torch, dgl
from models.tri_explainer import TriExplainer
g = dgl.graph(([0,1,2,3,0,2],[1,2,3,0,3,1]))
g.edata['edge_type'] = torch.tensor([0,1,2,0,1,2])
features = torch.randn(4, 64)
preds = torch.tensor([0,1,0,1])
probs = torch.tensor([[0.9,0.1],[0.2,0.8],[0.85,0.15],[0.3,0.7]])
expl_dict = {'gate_weights': torch.tensor([0.1,0.2,0.15,0.12,0.08,0.1,0.15,0.1]),
             'relation_weights': torch.tensor([0.3, 0.5, 0.2])}
explainer = TriExplainer(top_k_features=3, top_k_edges=3)
result = explainer.explain_node(1, expl_dict, g, features, preds, probs)
assert 'feature_attribution' in result
assert 'edge_attribution' in result
assert 'subgraph_attribution' in result
print('TriExplainer OK:', result['subgraph_attribution'])
"
```

- [ ] **Step 3: Commit**

```bash
git add models/tri_explainer.py
git commit -m "feat: implement TriExplainer three-level attribution (Phase 4)"
```

---

## Phase 1-4 Integration: Full TP-THGN Model

### Task INT.1: Assemble TP-THGN model

**Files:**
- Create: `models/tp_thgn.py`

- [ ] **Step 1: Create integrated model**

```python
"""
TP-THGN: Topology-Preserving Temporal Heterogeneous Graph Network
Integrates TD-GRU-GNN, TP-GraphSMOTE, XAttention, and TriExplainer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .td_gru_gnn import TD_GRU_GNN
from .tp_graphsmote import TPGraphSMOTE
from .xattention import XMultiLayerAttention
from .tri_explainer import TriExplainer


class TP_THGN(nn.Module):
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

        self.td_gru_gnn = TD_GRU_GNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2
        )

        self.tp_graphsmote = TPGraphSMOTE(
            embedding_dim=hidden_dim,
            k_neighbors=5,
            beta=beta_laplacian
        )

        self.xattention = XMultiLayerAttention(
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

    def forward(self, g, node_features, labels=None, timestamps=None,
                adj_matrices=None, training=True):
        """
        Returns:
            logits, embeddings, losses_dict
        """
        h = self.feature_extractor(node_features)
        h = self.td_gru_gnn(g, h, timestamps)

        # Oversampling (training only)
        lap_loss = torch.tensor(0.0, device=h.device)
        labels_processed = labels
        graph_changed = False

        if training and labels is not None and self.oversample_ratio > 1.0:
            num_fraud = (labels == 1).sum().item()
            if num_fraud > 0:
                adj = adj_matrices[0] if adj_matrices else torch.eye(h.shape[0], device=h.device)
                h_os, labels_os, adj_os, lap_loss = self.tp_graphsmote(
                    h, labels, adj, self.oversample_ratio
                )
                if h_os.shape[0] != h.shape[0]:
                    graph_changed = True
                h = h_os
                labels_processed = labels_os

        # Attention
        if not graph_changed:
            h = self.xattention(g, h, adj_matrices)

        h = self.dropout(h)
        logits = self.classifier(h)

        # Loss
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

    def predict(self, g, node_features, timestamps=None, adj_matrices=None):
        self.eval()
        with torch.no_grad():
            logits, emb, _ = self.forward(
                g, node_features, labels=None, timestamps=timestamps,
                adj_matrices=adj_matrices, training=False
            )
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

    def predict_with_explanation(self, g, node_features, timestamps=None, adj_matrices=None):
        """Predict and return attention weights for explanation."""
        self.eval()
        with torch.no_grad():
            h = self.feature_extractor(node_features)
            h = self.td_gru_gnn(g, h, timestamps)
            h, explanation = self.xattention(g, h, adj_matrices, explain=True)
            h = self.dropout(h)
            logits = self.classifier(h)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs, explanation


def create_tp_thgn_model(config):
    return TP_THGN(
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
```

- [ ] **Step 2: Verify full model forward pass**

```bash
conda activate thg-oafn
python -c "
import torch, dgl
from models.tp_thgn import create_tp_thgn_model
g = dgl.graph(([0,1,2,3,4,5,6],[1,2,3,4,5,6,0]))
g.edata['edge_type'] = torch.zeros(7, dtype=torch.long)
x = torch.randn(7, 25)
labels = torch.tensor([0,0,0,0,0,1,1])
ts = torch.linspace(0, 1, 7)
config = {'input_dim': 25, 'hidden_dim': 64, 'num_relations': 1, 'num_heads': 8, 'oversample_ratio': 2.0}
model = create_tp_thgn_model(config)
logits, emb, losses = model(g, x, labels, ts, training=True)
print(f'Logits: {logits.shape}, Loss: {losses[\"total_loss\"].item():.4f}')
preds, probs, expl = model.predict_with_explanation(g, x, ts)
print(f'Preds: {preds}, Explanation keys: {list(expl.keys())}')
print('TP-THGN full model OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add models/tp_thgn.py
git commit -m "feat: assemble full TP-THGN model integrating all modules"
```

### Task INT.2: Create TP-THGN training script

**Files:**
- Create: `train_tp_thgn.py`

- [ ] **Step 1: Create training script**

```python
"""Training script for TP-THGN model."""
import argparse
import json
import os
import torch
import torch.optim as optim
import numpy as np

from models.tp_thgn import create_tp_thgn_model
from utils.data_loader import load_real_dataset, split_data
from utils.metrics import calculate_metrics, print_metrics


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Device: {device}")

    dataset = load_real_dataset(args.dataset, data_dir=args.data_dir)
    g = dataset.build_dgl_graph()
    features = torch.FloatTensor(dataset.node_features).to(device)
    labels = torch.LongTensor(dataset.labels).to(device)
    g = g.to(device)

    timestamps = None
    if dataset.timestamps is not None:
        timestamps = torch.FloatTensor(dataset.timestamps).to(device)

    train_mask, val_mask, test_mask = split_data(dataset)

    print(f"Nodes: {dataset.num_nodes}, Features: {dataset.num_features}")
    print(f"Fraud ratio: {(labels == 1).sum().item() / len(labels):.2%}")
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    config = {
        'input_dim': dataset.num_features,
        'hidden_dim': args.hidden_dim,
        'output_dim': 2,
        'num_relations': len(np.unique(dataset.edge_types)),
        'num_heads': args.num_heads,
        'num_hops': args.num_hops,
        'dropout': args.dropout,
        'oversample_ratio': args.oversample_ratio,
        'beta_laplacian': args.beta_laplacian,
    }

    model = create_tp_thgn_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits, emb, losses = model(g, features, labels, timestamps, training=True)
        loss = losses['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                preds, probs = model.predict(g, features, timestamps)
            y_true = labels[val_mask].cpu().numpy()
            y_pred = preds[val_mask].cpu().numpy()
            y_prob = probs[val_mask, 1].cpu().numpy()
            val_m = calculate_metrics(y_true, y_pred, y_prob)

            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f} | "
                  f"Val F1: {val_m['F1']:.4f} | Val AUC: {val_m['AUC']:.4f}")

            if val_m['F1'] > best_val_f1:
                best_val_f1 = val_m['F1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds, probs = model.predict(g, features, timestamps)
    y_true = labels[test_mask].cpu().numpy()
    y_pred = preds[test_mask].cpu().numpy()
    y_prob = probs[test_mask, 1].cpu().numpy()
    test_m = calculate_metrics(y_true, y_pred, y_prob)
    print_metrics(test_m, "Test")

    # Save
    os.makedirs('experiments/results', exist_ok=True)
    result = {
        'model': 'TP-THGN',
        'dataset': args.dataset,
        'seed': args.seed,
        'config': config,
        'test_metrics': {k: v for k, v in test_m.items() if k != 'Confusion_Matrix'},
    }
    out_path = f'experiments/results/tp_thgn_{args.dataset}_seed{args.seed}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(best_state, f'checkpoints/tp_thgn_{args.dataset}_best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Amazon')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--oversample_ratio', type=float, default=2.0)
    parser.add_argument('--beta_laplacian', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()
    train(args)
```

- [ ] **Step 2: Run a quick training test**

```bash
conda activate thg-oafn
python train_tp_thgn.py --dataset Amazon --data_dir ./data --epochs 30 --use_gpu --eval_every 5
```

Expected: Training completes, prints test metrics, saves JSON result.

- [ ] **Step 3: Commit**

```bash
git add train_tp_thgn.py
git commit -m "feat: add TP-THGN training script with full pipeline"
```

---

## Phase 5: Ablation & Comparison Experiments

### Task 5.1: Create baselines

**Files:**
- Create: `models/baselines/__init__.py`
- Create: `models/baselines/gcn_baseline.py`
- Create: `models/baselines/gat_baseline.py`
- Create: `models/baselines/graphsage_baseline.py`
- Create: `models/baselines/xgboost_baseline.py`
- Create: `models/baselines/lr_baseline.py`

- [ ] **Step 1: Create GNN baselines (GCN, GAT, GraphSAGE)**

`models/baselines/__init__.py`:
```python
from .gcn_baseline import GCNBaseline
from .gat_baseline import GATBaseline
from .graphsage_baseline import GraphSAGEBaseline
```

`models/baselines/gcn_baseline.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            return F.relu(self.W(g.ndata['h_neigh']))


class GCNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
            h = self.dropout(h)
        return self.classifier(h)

    def predict(self, g, features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
```

`models/baselines/gat_baseline.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv


class GATBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout))
        self.classifier = nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h).flatten(1)
            h = F.elu(h)
        return self.classifier(h)

    def predict(self, g, features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
```

`models/baselines/graphsage_baseline.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class GraphSAGEBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dim, 'mean'))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, 'mean'))
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = F.relu(layer(g, h))
            h = self.dropout(h)
        return self.classifier(h)

    def predict(self, g, features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
```

`models/baselines/xgboost_baseline.py`:
```python
from xgboost import XGBClassifier
import numpy as np


class XGBoostBaseline:
    def __init__(self, scale_pos_weight=10):
        self.model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss',
            use_label_encoder=False
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs
```

`models/baselines/lr_baseline.py`:
```python
from sklearn.linear_model import LogisticRegression
import numpy as np


class LRBaseline:
    def __init__(self, class_weight='balanced'):
        self.model = LogisticRegression(
            max_iter=1000, class_weight=class_weight, solver='lbfgs'
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs
```

- [ ] **Step 2: Commit baselines**

```bash
git add models/baselines/
git commit -m "feat: add baseline models (GCN, GAT, GraphSAGE, XGBoost, LR)"
```

### Task 5.2: Create ablation experiment script

**Files:**
- Create: `experiments/run_ablation.py`

- [ ] **Step 1: Create ablation runner**

```python
"""
Ablation study: test each module's contribution.
Variants:
  - Full TP-THGN (all modules)
  - w/o Time Decay (replace TD-GRU-GNN with original GRU-GNN)
  - w/o Laplacian (beta=0, standard GraphSMOTE)
  - w/o Oversampling (ratio=1.0)
  - w/o XAttention (replace with original attention)
"""
import json
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tp_thgn import create_tp_thgn_model
from utils.data_loader import load_real_dataset, split_data
from utils.metrics import calculate_metrics


def run_variant(config, dataset_name, seeds, variant_name):
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = load_real_dataset(dataset_name, data_dir='./data')
        g = dataset.build_dgl_graph()
        features = torch.FloatTensor(dataset.node_features)
        labels = torch.LongTensor(dataset.labels)
        train_mask, val_mask, test_mask = split_data(dataset)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features, labels, g = features.to(device), labels.to(device), g.to(device)

        timestamps = None
        if dataset.timestamps is not None:
            timestamps = torch.FloatTensor(dataset.timestamps).to(device)

        model = create_tp_thgn_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_val_f1, best_state, patience = 0.0, None, 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits, emb, losses = model(g, features, labels, timestamps, training=True)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    preds, probs = model.predict(g, features, timestamps)
                val_m = calculate_metrics(
                    labels[val_mask].cpu().numpy(),
                    preds[val_mask].cpu().numpy(),
                    probs[val_mask, 1].cpu().numpy()
                )
                if val_m['F1'] > best_val_f1:
                    best_val_f1 = val_m['F1']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 10:
                    break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            preds, probs = model.predict(g, features, timestamps)
        test_m = calculate_metrics(
            labels[test_mask].cpu().numpy(),
            preds[test_mask].cpu().numpy(),
            probs[test_mask, 1].cpu().numpy()
        )
        all_metrics.append(test_m)

    result = {'variant': variant_name}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def main():
    dataset_name = 'Amazon'
    seeds = [42, 123, 456, 789, 2024]

    dataset = load_real_dataset(dataset_name, data_dir='./data')
    base_config = {
        'input_dim': dataset.num_features,
        'hidden_dim': 64, 'output_dim': 2,
        'num_relations': len(np.unique(dataset.edge_types)),
        'num_heads': 8, 'num_hops': 2, 'dropout': 0.5,
        'oversample_ratio': 2.0, 'beta_laplacian': 0.01,
    }

    variants = {
        'Full TP-THGN': base_config.copy(),
        'w/o Laplacian': {**base_config, 'beta_laplacian': 0.0},
        'w/o Oversampling': {**base_config, 'oversample_ratio': 1.0},
        'w/o Both (no OS, no Lap)': {**base_config, 'oversample_ratio': 1.0, 'beta_laplacian': 0.0},
    }

    results = []
    for name, config in variants.items():
        print(f"\n{'='*50}\nRunning: {name}\n{'='*50}")
        r = run_variant(config, dataset_name, seeds, name)
        results.append(r)
        print(f"  F1: {r['F1']['mean']:.4f} ± {r['F1']['std']:.4f}")

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nAblation results saved.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run ablation**

```bash
conda activate thg-oafn
python experiments/run_ablation.py
```

- [ ] **Step 3: Commit**

```bash
git add experiments/run_ablation.py experiments/results/ablation_results.json
git commit -m "feat: run ablation study, save results (Phase 5)"
```

### Task 5.3: Create comparison experiment script

**Files:**
- Create: `experiments/run_comparison.py`

- [ ] **Step 1: Create comparison runner**

```python
"""Run all baselines and TP-THGN for comparison table."""
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tp_thgn import create_tp_thgn_model
from models.baselines.gcn_baseline import GCNBaseline
from models.baselines.gat_baseline import GATBaseline
from models.baselines.graphsage_baseline import GraphSAGEBaseline
from models.baselines.xgboost_baseline import XGBoostBaseline
from models.baselines.lr_baseline import LRBaseline
from models.thg_oafn import create_thg_oafn_model
from utils.data_loader import load_real_dataset, split_data
from utils.metrics import calculate_metrics


def train_gnn_baseline(ModelClass, model_kwargs, dataset_name, seeds):
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset = load_real_dataset(dataset_name, data_dir='./data')
        g = dataset.build_dgl_graph()
        features = torch.FloatTensor(dataset.node_features)
        labels = torch.LongTensor(dataset.labels)
        train_mask, val_mask, test_mask = split_data(dataset)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features, labels, g = features.to(device), labels.to(device), g.to(device)

        model = ModelClass(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        # Class weights
        num_fraud = (labels == 1).sum().float()
        num_normal = (labels == 0).sum().float()
        w = torch.clamp((num_normal / num_fraud).sqrt(), 1.5, 5.0).item()
        class_w = torch.tensor([1.0, w], device=device)

        best_f1, best_state, patience = 0.0, None, 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits = model(g, features)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_w)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                preds, probs = model.predict(g, features)
                vm = calculate_metrics(labels[val_mask].cpu().numpy(), preds[val_mask].cpu().numpy(), probs[val_mask, 1].cpu().numpy())
                if vm['F1'] > best_f1:
                    best_f1 = vm['F1']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 10:
                    break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        preds, probs = model.predict(g, features)
        tm = calculate_metrics(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), probs[test_mask, 1].cpu().numpy())
        all_metrics.append(tm)

    result = {}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def train_sklearn_baseline(ModelClass, dataset_name, seeds):
    all_metrics = []
    for seed in seeds:
        np.random.seed(seed)
        dataset = load_real_dataset(dataset_name, data_dir='./data')
        train_mask, val_mask, test_mask = split_data(dataset)
        X_train = dataset.node_features[train_mask]
        y_train = dataset.labels[train_mask]
        X_test = dataset.node_features[test_mask]
        y_test = dataset.labels[test_mask]

        model = ModelClass()
        model.fit(X_train, y_train)
        preds, probs = model.predict(X_test)
        tm = calculate_metrics(y_test, preds, probs[:, 1])
        all_metrics.append(tm)

    result = {}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def main():
    dataset_name = 'Amazon'
    seeds = [42, 123, 456, 789, 2024]
    dataset = load_real_dataset(dataset_name, data_dir='./data')
    input_dim = dataset.num_features

    results = []

    # LR
    print("Running LR...")
    r = train_sklearn_baseline(LRBaseline, dataset_name, seeds)
    r['model'] = 'Logistic Regression'
    results.append(r)

    # XGBoost
    print("Running XGBoost...")
    r = train_sklearn_baseline(XGBoostBaseline, dataset_name, seeds)
    r['model'] = 'XGBoost'
    results.append(r)

    # GCN
    print("Running GCN...")
    r = train_gnn_baseline(GCNBaseline, {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2}, dataset_name, seeds)
    r['model'] = 'GCN'
    results.append(r)

    # GAT
    print("Running GAT...")
    r = train_gnn_baseline(GATBaseline, {'input_dim': input_dim, 'hidden_dim': 16, 'output_dim': 2, 'num_heads': 4}, dataset_name, seeds)
    r['model'] = 'GAT'
    results.append(r)

    # GraphSAGE
    print("Running GraphSAGE...")
    r = train_gnn_baseline(GraphSAGEBaseline, {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2}, dataset_name, seeds)
    r['model'] = 'GraphSAGE'
    results.append(r)

    # Print summary
    print(f"\n{'Model':<20} {'AUC':>10} {'F1':>10} {'Recall':>10} {'Precision':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} {r['AUC']['mean']:>10.4f} {r['F1']['mean']:>10.4f} "
              f"{r['Recall']['mean']:>10.4f} {r['Precision']['mean']:>10.4f}")

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run comparison**

```bash
conda activate thg-oafn
python experiments/run_comparison.py
```

- [ ] **Step 3: Commit**

```bash
git add experiments/run_comparison.py experiments/results/comparison_results.json
git commit -m "feat: run comparison experiments against all baselines (Phase 5)"
```

---

## Phase 6: Cross-Dataset Validation & Visualization

### Task 6.1: Create visualization utilities

**Files:**
- Create: `utils/visualization.py`

- [ ] **Step 1: Create paper figure generation module**

```python
"""Visualization utilities for paper figures."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150


def plot_ablation_bar(results_path, output_path='figures/ablation_bar.pdf'):
    """Fig.3: Ablation study bar chart."""
    with open(results_path) as f:
        results = json.load(f)

    variants = [r['variant'] for r in results]
    metrics = ['F1', 'AUC', 'Recall', 'AUPRC']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variants))
    width = 0.2

    for i, metric in enumerate(metrics):
        means = [r[metric]['mean'] for r in results]
        stds = [r[metric]['std'] for r in results]
        ax.bar(x + i * width, means, width, yerr=stds, label=metric, capsize=3)

    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Score')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison_table(results_path, output_path='figures/comparison_table.pdf'):
    """Tab.3: Comparison table as figure."""
    with open(results_path) as f:
        results = json.load(f)

    models = [r['model'] for r in results]
    metrics = ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC']

    cell_text = []
    for r in results:
        row = []
        for m in metrics:
            mean = r[m]['mean']
            std = r[m]['std']
            row.append(f"{mean:.4f}±{std:.4f}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(12, len(models) * 0.6 + 1))
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=models, colLabels=metrics,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_curve(log_path, output_path='figures/training_curve.pdf'):
    """Fig.2: Training convergence curve."""
    with open(log_path) as f:
        log = json.load(f)

    epochs = log['epochs']
    losses = log['losses']
    val_f1s = log['val_f1s']

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, losses, 'r-', alpha=0.7, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1-Score', color='tab:blue')
    ax2.plot(epochs, val_f1s, 'b-', alpha=0.7, label='Val F1')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_attention_heatmap(gate_weights, relation_weights, output_path='figures/attention_heatmap.pdf'):
    """Fig.4: Attention weight heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Gate weights
    gw = np.array(gate_weights).reshape(1, -1)
    sns.heatmap(gw, ax=axes[0], annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'Head {i+1}' for i in range(len(gate_weights))],
                yticklabels=['Weight'])
    axes[0].set_title('Information Perception Gate Weights')

    # Relation weights
    rw = np.array(relation_weights).reshape(1, -1)
    rel_names = ['UPU', 'USU', 'UVU'][:len(relation_weights)]
    sns.heatmap(rw, ax=axes[1], annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=rel_names, yticklabels=['Weight'])
    axes[1].set_title('Relation Fusion Weights')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
```

- [ ] **Step 2: Verify visualization**

```bash
conda activate thg-oafn
python -c "
from utils.visualization import plot_attention_heatmap
plot_attention_heatmap([0.12, 0.15, 0.13, 0.11, 0.14, 0.12, 0.11, 0.12], [0.4, 0.35, 0.25])
print('Visualization OK')
"
```

Expected: Creates `figures/attention_heatmap.pdf`.

- [ ] **Step 3: Commit**

```bash
git add utils/visualization.py figures/
git commit -m "feat: add visualization utilities for paper figures (Phase 6)"
```

### Task 6.2: Generate all paper figures

**Files:**
- Create: `experiments/generate_figures.py`

- [ ] **Step 1: Create figure generation script**

```python
"""Generate all paper figures from experiment results."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import (
    plot_ablation_bar,
    plot_comparison_table,
    plot_attention_heatmap,
)


def main():
    # Fig.3: Ablation bar chart
    if os.path.exists('experiments/results/ablation_results.json'):
        plot_ablation_bar('experiments/results/ablation_results.json')

    # Tab.3: Comparison table
    if os.path.exists('experiments/results/comparison_results.json'):
        plot_comparison_table('experiments/results/comparison_results.json')

    print("\nAll available figures generated in figures/")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run figure generation**

```bash
conda activate thg-oafn
python experiments/generate_figures.py
```

- [ ] **Step 3: Commit all figures**

```bash
git add experiments/generate_figures.py figures/
git commit -m "feat: generate paper figures from experiment results (Phase 6)"
```

---

## Summary: Execution Order

1. Phase 0: Tasks 0.1 → 0.2 → 0.3 → 0.4
2. Phase 1: Task 1.1
3. Phase 2: Task 2.1
4. Phase 3: Task 3.1
5. Phase 4: Task 4.1
6. Integration: Tasks INT.1 → INT.2
7. Phase 5: Tasks 5.1 → 5.2 → 5.3
8. Phase 6: Tasks 6.1 → 6.2

Each phase creates a branch `experiment/phaseN-*`, runs experiments, commits results, then merges code back to main.
