"""
Ablation study: test each module's contribution.
Variants:
  - Full TP-THGN (all modules)
  - w/o Time Decay (timestamps=None forced)
  - w/o Laplacian (beta=0)
  - w/o Oversampling (ratio=1.0)
  - w/o XAttention (skip attention layer)
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


def run_variant(config, dataset_name, seeds, variant_name, force_no_timestamps=False):
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = load_real_dataset(dataset_name, data_dir='./data')
        g = dataset.build_dgl_graph()
        features = torch.FloatTensor(dataset.node_features)
        labels = torch.LongTensor(dataset.labels)
        train_mask, val_mask, test_mask = split_data(dataset)

        timestamps = None
        if dataset.timestamps is not None and not force_no_timestamps:
            timestamps = torch.FloatTensor(dataset.timestamps)

        model = create_tp_thgn_model(config)
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
            preds, probs = model.predict(g, features, timestamps)
        test_m = calculate_metrics(
            labels[test_mask].cpu().numpy(),
            preds[test_mask].cpu().numpy(),
            probs[test_mask, 1].cpu().numpy()
        )
        all_metrics.append(test_m)
        print(f"    Seed {seed}: F1={test_m['F1']:.4f}")

    result = {'variant': variant_name}
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        vals = [m.get(key, 0) for m in all_metrics]
        result[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return result


def main():
    dataset_name = 'Amazon'
    seeds = [42, 123, 456]

    dataset = load_real_dataset(dataset_name, data_dir='./data')
    base_config = {
        'input_dim': dataset.num_features,
        'hidden_dim': 64, 'output_dim': 2,
        'num_relations': len(np.unique(dataset.edge_types)),
        'num_heads': 8, 'num_hops': 2, 'dropout': 0.5,
        'oversample_ratio': 2.0, 'beta_laplacian': 0.01,
    }

    variants = [
        ('Full TP-THGN', base_config.copy(), False),
        ('w/o Time Decay', base_config.copy(), True),
        ('w/o Laplacian', {**base_config, 'beta_laplacian': 0.0}, False),
        ('w/o Oversampling', {**base_config, 'oversample_ratio': 1.0}, False),
        ('w/o OS + Lap', {**base_config, 'oversample_ratio': 1.0, 'beta_laplacian': 0.0}, False),
    ]

    results = []
    for name, config, no_ts in variants:
        print(f"\n{'='*50}\nRunning: {name}\n{'='*50}")
        r = run_variant(config, dataset_name, seeds, name, force_no_timestamps=no_ts)
        results.append(r)
        print(f"  => F1: {r['F1']['mean']:.4f} +/- {r['F1']['std']:.4f}")

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{'Variant':<20} {'F1':>12} {'AUC':>12} {'Recall':>12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['variant']:<20} {r['F1']['mean']:>5.4f}±{r['F1']['std']:.4f} "
              f"{r['AUC']['mean']:>5.4f}±{r['AUC']['std']:.4f} "
              f"{r['Recall']['mean']:>5.4f}±{r['Recall']['std']:.4f}")
    print(f"\nAblation results saved to experiments/results/ablation_results.json")


if __name__ == '__main__':
    main()
