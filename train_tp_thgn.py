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
    training_log = {'epochs': [], 'losses': [], 'val_f1s': [], 'val_aucs': []}

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

            training_log['epochs'].append(epoch + 1)
            training_log['losses'].append(loss.item())
            training_log['val_f1s'].append(val_m['F1'])
            training_log['val_aucs'].append(val_m['AUC'])

            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f} | "
                  f"Val F1: {val_m['F1']:.4f} | Val AUC: {val_m['AUC']:.4f} | "
                  f"Val Recall: {val_m['Recall']:.4f}")

            if val_m['F1'] > best_val_f1:
                best_val_f1 = val_m['F1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds, probs = model.predict(g, features, timestamps)
    y_true = labels[test_mask].cpu().numpy()
    y_pred = preds[test_mask].cpu().numpy()
    y_prob = probs[test_mask, 1].cpu().numpy()
    test_m = calculate_metrics(y_true, y_pred, y_prob)
    print_metrics(test_m, "Test")

    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    result = {
        'model': 'TP-THGN',
        'dataset': args.dataset,
        'seed': args.seed,
        'config': config,
        'test_metrics': {k: v for k, v in test_m.items() if k != 'Confusion_Matrix'},
        'training_log': training_log,
    }
    out_path = f'experiments/results/tp_thgn_{args.dataset}_seed{args.seed}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(best_state, f'checkpoints/tp_thgn_{args.dataset}_best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TP-THGN model')
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
