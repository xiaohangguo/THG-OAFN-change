"""Run baseline models and save results."""
import json
import os
import sys
import torch
import numpy as np
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.thg_oafn import create_thg_oafn_model
from utils.data_loader import load_real_dataset, split_data
from utils.metrics import calculate_metrics, print_metrics


def run_thg_oafn_baseline(dataset_name='Amazon', seeds=None):
    if seeds is None:
        seeds = [42, 123, 456, 789, 2024]

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
        best_state = None
        patience = 0

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits, embeddings, losses = model(g, features, labels, training=True)
            loss = losses['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
            preds, probs = model.predict(g, features)
        y_true = labels[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
        y_prob = probs[test_mask, 1].cpu().numpy()
        test_m = calculate_metrics(y_true, y_pred, y_prob)
        all_metrics.append(test_m)
        print(f"  Seed {seed}: F1={test_m['F1']:.4f} AUC={test_m['AUC']:.4f} "
              f"AUPRC={test_m.get('AUPRC', 0):.4f} Recall={test_m['Recall']:.4f}")

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

    print(f"\nTHG-OAFN Baseline Results (mean +/- std):")
    for key in ['AUC', 'F1', 'Recall', 'Precision', 'AUPRC', 'G-Mean']:
        print(f"  {key}: {result[key]['mean']:.4f} +/- {result[key]['std']:.4f}")

    return result


if __name__ == '__main__':
    run_thg_oafn_baseline()
