"""
GPU-compatible experiment runner.
Runs all experiments (baseline, ablation, comparison) on GPU.
"""
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tp_thgn_gpu import create_tp_thgn_gpu_model
from models.baselines.gcn_baseline import GCNBaseline
from models.baselines.gat_baseline import GATBaseline
from models.baselines.graphsage_baseline import GraphSAGEBaseline
from models.baselines.xgboost_baseline import XGBoostBaseline
from models.baselines.lr_baseline import LRBaseline
from utils.data_loader import load_real_dataset, split_data
from utils.graph_utils import build_sparse_adj
from utils.metrics import calculate_metrics


def run_tp_thgn_gpu(config, dataset_name, seeds, variant_name, data_dir='./data'):
    """Run TP-THGN GPU variant with multiple seeds."""
    all_metrics = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = load_real_dataset(dataset_name, data_dir=data_dir)
        adj, adj_per_type = build_sparse_adj(
            dataset.edge_index, dataset.num_nodes,
            edge_types=dataset.edge_types, normalize=True
        )
        adj = adj.to(device)
        if adj_per_type:
            adj_per_type = {k: v.to(device) for k, v in adj_per_type.items()}

        features = torch.FloatTensor(dataset.node_features).to(device)
        labels = torch.LongTensor(dataset.labels).to(device)
        train_mask, val_mask, test_mask = split_data(dataset)

        timestamps = None
        if dataset.timestamps is not None:
            timestamps = torch.FloatTensor(dataset.timestamps).to(device)

        model = create_tp_thgn_gpu_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_val_f1, best_state, patience = 0.0, None, 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits, emb, losses = model(adj, features, labels, timestamps,
                                        adj_per_type=adj_per_type, training=True)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    preds, probs = model.predict(adj, features, timestamps, adj_per_type)
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

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            preds, probs = model.predict(adj, features, timestamps, adj_per_type)
        test_m = calculate_metrics(
            labels[test_mask].cpu().numpy(),
            preds[test_mask].cpu().numpy(),
            probs[test_mask, 1].cpu().numpy()
        )
        all_metrics.append(test_m)
        print(f"  Seed {seed}: F1={test_m['F1']:.4f} AUC={test_m['AUC']:.4f}")

    result = {'variant': variant_name}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def run_gnn_baseline_gpu(ModelClass, model_kwargs, dataset_name, seeds, data_dir='./data'):
    """Run GNN baseline on GPU using DGL-free sparse ops."""
    all_metrics = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset = load_real_dataset(dataset_name, data_dir=data_dir)
        adj, _ = build_sparse_adj(dataset.edge_index, dataset.num_nodes, normalize=True)
        adj = adj.to(device)
        features = torch.FloatTensor(dataset.node_features).to(device)
        labels = torch.LongTensor(dataset.labels).to(device)
        train_mask, val_mask, test_mask = split_data(dataset)

        # GNN baselines use DGL - run on CPU with DGL graph
        import dgl
        g = dataset.build_dgl_graph()
        features_cpu = torch.FloatTensor(dataset.node_features)
        labels_cpu = torch.LongTensor(dataset.labels)

        model = ModelClass(**model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        num_fraud = (labels_cpu == 1).sum().float()
        num_normal = (labels_cpu == 0).sum().float()
        w = torch.clamp((num_normal / num_fraud).sqrt(), 1.5, 5.0).item()
        class_w = torch.tensor([1.0, w])

        best_f1, best_state, patience = 0.0, None, 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits = model(g, features_cpu)
            loss = F.cross_entropy(logits[train_mask], labels_cpu[train_mask], weight=class_w)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                preds, probs = model.predict(g, features_cpu)
                vm = calculate_metrics(
                    labels_cpu[val_mask].numpy(),
                    preds[val_mask].numpy(),
                    probs[val_mask, 1].numpy()
                )
                if vm['F1'] > best_f1:
                    best_f1 = vm['F1']
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 10:
                    break

        if best_state is None:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        preds, probs = model.predict(g, features_cpu)
        tm = calculate_metrics(
            labels_cpu[test_mask].numpy(),
            preds[test_mask].numpy(),
            probs[test_mask, 1].numpy()
        )
        all_metrics.append(tm)

    result = {}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def run_sklearn_baseline(ModelClass, dataset_name, seeds, data_dir='./data'):
    """Run sklearn baseline."""
    all_metrics = []
    for seed in seeds:
        np.random.seed(seed)
        dataset = load_real_dataset(dataset_name, data_dir=data_dir)
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
    data_dir = './data'
    seeds = [42, 123, 456, 789, 2024]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running experiments on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = load_real_dataset(dataset_name, data_dir=data_dir)
    input_dim = dataset.num_features
    num_relations = len(np.unique(dataset.edge_types))

    base_config = {
        'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2,
        'num_relations': num_relations, 'num_heads': 8, 'num_hops': 2,
        'dropout': 0.5, 'oversample_ratio': 2.0, 'beta_laplacian': 0.01,
    }

    os.makedirs('experiments/results', exist_ok=True)
    all_results = []

    # === Part 1: Sklearn Baselines ===
    print("\n" + "="*60)
    print("Part 1: Sklearn Baselines")
    print("="*60)

    print("\nRunning Logistic Regression...")
    r = run_sklearn_baseline(LRBaseline, dataset_name, seeds, data_dir)
    r['model'] = 'Logistic Regression'
    all_results.append(r)
    print(f"  F1: {r['F1']['mean']:.4f} +/- {r['F1']['std']:.4f}")

    print("\nRunning XGBoost...")
    r = run_sklearn_baseline(XGBoostBaseline, dataset_name, seeds, data_dir)
    r['model'] = 'XGBoost'
    all_results.append(r)
    print(f"  F1: {r['F1']['mean']:.4f} +/- {r['F1']['std']:.4f}")

    # === Part 2: GNN Baselines (CPU with DGL) ===
    print("\n" + "="*60)
    print("Part 2: GNN Baselines")
    print("="*60)

    print("\nRunning GCN...")
    r = run_gnn_baseline_gpu(GCNBaseline,
                             {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2},
                             dataset_name, seeds, data_dir)
    r['model'] = 'GCN'
    all_results.append(r)
    print(f"  F1: {r['F1']['mean']:.4f} +/- {r['F1']['std']:.4f}")

    print("\nRunning GAT...")
    r = run_gnn_baseline_gpu(GATBaseline,
                             {'input_dim': input_dim, 'hidden_dim': 16, 'output_dim': 2, 'num_heads': 4},
                             dataset_name, seeds, data_dir)
    r['model'] = 'GAT'
    all_results.append(r)
    print(f"  F1: {r['F1']['mean']:.4f} +/- {r['F1']['std']:.4f}")

    print("\nRunning GraphSAGE...")
    r = run_gnn_baseline_gpu(GraphSAGEBaseline,
                             {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2},
                             dataset_name, seeds, data_dir)
    r['model'] = 'GraphSAGE'
    all_results.append(r)
    print(f"  F1: {r['F1']['mean']:.4f} +/- {r['F1']['std']:.4f}")

    # Save comparison results
    with open('experiments/results/comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nComparison results saved.")

    # === Part 3: TP-THGN Ablation (GPU) ===
    print("\n" + "="*60)
    print("Part 3: TP-THGN Ablation Study (GPU)")
    print("="*60)

    ablation_results = []
    variants = {
        'Full TP-THGN': base_config.copy(),
        'w/o Laplacian': {**base_config, 'beta_laplacian': 0.0},
        'w/o Oversampling': {**base_config, 'oversample_ratio': 1.0},
        'w/o Both': {**base_config, 'oversample_ratio': 1.0, 'beta_laplacian': 0.0},
    }

    for name, config in variants.items():
        print(f"\nRunning: {name}...")
        start = time.time()
        r = run_tp_thgn_gpu(config, dataset_name, seeds, name, data_dir)
        elapsed = time.time() - start
        ablation_results.append(r)
        print(f"  F1: {r['F1']['mean']:.4f} +/- {r['F1']['std']:.4f} ({elapsed:.1f}s)")

    with open('experiments/results/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print("\nAblation results saved.")

    # Add TP-THGN to comparison
    tp_thgn_result = ablation_results[0].copy()
    tp_thgn_result['model'] = 'TP-THGN (ours)'
    all_results.append(tp_thgn_result)
    with open('experiments/results/comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # === Summary ===
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(f"\n{'Model':<22} {'AUC':>8} {'F1':>8} {'Recall':>8} {'Prec':>8} {'AUPRC':>8}")
    print("-" * 70)
    for r in all_results:
        name = r.get('model', r.get('variant', '?'))
        print(f"{name:<22} {r['AUC']['mean']:>8.4f} {r['F1']['mean']:>8.4f} "
              f"{r['Recall']['mean']:>8.4f} {r['Precision']['mean']:>8.4f} "
              f"{r.get('AUPRC', {}).get('mean', 0):>8.4f}")

    if device.type == 'cuda':
        print(f"\nPeak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")


if __name__ == '__main__':
    main()
