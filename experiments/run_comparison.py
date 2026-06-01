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

        model = ModelClass(**model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        num_fraud = (labels == 1).sum().float()
        num_normal = (labels == 0).sum().float()
        w = torch.clamp((num_normal / num_fraud).sqrt(), 1.5, 5.0).item()
        class_w = torch.tensor([1.0, w])

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
                vm = calculate_metrics(
                    labels[val_mask].numpy(), preds[val_mask].numpy(),
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
        preds, probs = model.predict(g, features)
        tm = calculate_metrics(
            labels[test_mask].numpy(), preds[test_mask].numpy(),
            probs[test_mask, 1].numpy()
        )
        all_metrics.append(tm)
        print(f"    Seed {seed}: F1={tm['F1']:.4f}")

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
        print(f"    Seed {seed}: F1={tm['F1']:.4f}")

    result = {}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def train_tp_thgn(dataset_name, seeds):
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset = load_real_dataset(dataset_name, data_dir='./data')
        g = dataset.build_dgl_graph()
        features = torch.FloatTensor(dataset.node_features)
        labels = torch.LongTensor(dataset.labels)
        train_mask, val_mask, test_mask = split_data(dataset)

        config = {
            'input_dim': dataset.num_features, 'hidden_dim': 64, 'output_dim': 2,
            'num_relations': len(np.unique(dataset.edge_types)),
            'num_heads': 8, 'num_hops': 2, 'dropout': 0.5,
            'oversample_ratio': 2.0, 'beta_laplacian': 0.01,
        }
        model = create_tp_thgn_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_f1, best_state, patience = 0.0, None, 0
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits, emb, losses = model(g, features, labels, training=True)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    preds, probs = model.predict(g, features)
                vm = calculate_metrics(
                    labels[val_mask].numpy(), preds[val_mask].numpy(),
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
        model.eval()
        with torch.no_grad():
            preds, probs = model.predict(g, features)
        tm = calculate_metrics(
            labels[test_mask].numpy(), preds[test_mask].numpy(),
            probs[test_mask, 1].numpy()
        )
        all_metrics.append(tm)
        print(f"    Seed {seed}: F1={tm['F1']:.4f}")

    result = {}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def main():
    dataset_name = 'Amazon'
    seeds = [42, 123, 456]
    dataset = load_real_dataset(dataset_name, data_dir='./data')
    input_dim = dataset.num_features

    results = []

    print("\n[1/7] Logistic Regression...")
    r = train_sklearn_baseline(LRBaseline, dataset_name, seeds)
    r['model'] = 'Logistic Regression'
    results.append(r)

    print("\n[2/7] XGBoost...")
    r = train_sklearn_baseline(XGBoostBaseline, dataset_name, seeds)
    r['model'] = 'XGBoost'
    results.append(r)

    print("\n[3/7] GCN...")
    r = train_gnn_baseline(GCNBaseline, {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2}, dataset_name, seeds)
    r['model'] = 'GCN'
    results.append(r)

    print("\n[4/7] GAT...")
    r = train_gnn_baseline(GATBaseline, {'input_dim': input_dim, 'hidden_dim': 16, 'output_dim': 2, 'num_heads': 4}, dataset_name, seeds)
    r['model'] = 'GAT'
    results.append(r)

    print("\n[5/7] GraphSAGE...")
    r = train_gnn_baseline(GraphSAGEBaseline, {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2}, dataset_name, seeds)
    r['model'] = 'GraphSAGE'
    results.append(r)

    print("\n[6/7] THG-OAFN (original)...")
    r = train_gnn_baseline(
        lambda **kw: create_thg_oafn_model({'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2, 'num_relations': len(np.unique(dataset.edge_types)), 'num_heads': 8, 'num_hops': 2, 'dropout': 0.5, 'oversample_ratio': 1.0}),
        {}, dataset_name, seeds
    )
    r['model'] = 'THG-OAFN (original)'
    results.append(r)

    print("\n[7/7] TP-THGN (ours)...")
    r = train_tp_thgn(dataset_name, seeds)
    r['model'] = 'TP-THGN (ours)'
    results.append(r)

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Model':<22} {'AUC':>12} {'F1':>12} {'Recall':>12} {'Precision':>12}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['model']:<22} {r['AUC']['mean']:>5.4f}±{r['AUC']['std']:.3f} "
              f"{r['F1']['mean']:>5.4f}±{r['F1']['std']:.3f} "
              f"{r['Recall']['mean']:>5.4f}±{r['Recall']['std']:.3f} "
              f"{r['Precision']['mean']:>5.4f}±{r['Precision']['std']:.3f}")

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison results saved.")


if __name__ == '__main__':
    main()
