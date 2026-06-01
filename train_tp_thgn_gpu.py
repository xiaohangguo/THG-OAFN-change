"""
GPU Training script for TP-THGN model.
Uses PyTorch sparse ops instead of DGL CUDA for full GPU acceleration.
Supports mixed precision training for RTX 4070.
"""
import argparse
import json
import os
import time
import torch
import torch.optim as optim
import numpy as np

from models.tp_thgn_gpu import create_tp_thgn_gpu_model
from utils.data_loader import load_real_dataset, split_data
from utils.graph_utils import build_sparse_adj
from utils.metrics import calculate_metrics, print_metrics


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Load dataset
    dataset = load_real_dataset(args.dataset, data_dir=args.data_dir)
    print(f"Nodes: {dataset.num_nodes}, Features: {dataset.num_features}")
    print(f"Edges: {dataset.num_edges}")
    print(f"Fraud ratio: {(dataset.labels == 1).sum() / len(dataset.labels):.2%}")

    # Build sparse adjacency matrix (GPU-compatible)
    print("Building sparse adjacency matrix...")
    adj, adj_per_type = build_sparse_adj(
        dataset.edge_index, dataset.num_nodes,
        edge_types=dataset.edge_types, normalize=True
    )

    # Move everything to GPU
    adj = adj.to(device)
    if adj_per_type is not None:
        adj_per_type = {k: v.to(device) for k, v in adj_per_type.items()}

    features = torch.FloatTensor(dataset.node_features).to(device)
    labels = torch.LongTensor(dataset.labels).to(device)

    timestamps = None
    if dataset.timestamps is not None:
        timestamps = torch.FloatTensor(dataset.timestamps).to(device)

    train_mask, val_mask, test_mask = split_data(dataset)
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    # Model config
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

    model = create_tp_thgn_gpu_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Mixed precision scaler for RTX 4070
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    training_log = {'epochs': [], 'losses': [], 'val_f1s': [], 'val_aucs': []}

    print(f"\nStarting training ({args.epochs} epochs)...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits, emb, losses = model(
                    adj, features, labels, timestamps,
                    adj_per_type=adj_per_type, training=True
                )
                loss = losses['total_loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, emb, losses = model(
                adj, features, labels, timestamps,
                adj_per_type=adj_per_type, training=True
            )
            loss = losses['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                preds, probs = model.predict(adj, features, timestamps, adj_per_type)
            y_true = labels[val_mask].cpu().numpy()
            y_pred = preds[val_mask].cpu().numpy()
            y_prob = probs[val_mask, 1].cpu().numpy()
            val_m = calculate_metrics(y_true, y_pred, y_prob)

            training_log['epochs'].append(epoch + 1)
            training_log['losses'].append(loss.item())
            training_log['val_f1s'].append(val_m['F1'])
            training_log['val_aucs'].append(val_m['AUC'])

            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {loss.item():.4f} | "
                  f"Val F1: {val_m['F1']:.4f} | Val AUC: {val_m['AUC']:.4f} | "
                  f"Time: {elapsed:.1f}s")

            if val_m['F1'] > best_val_f1:
                best_val_f1 = val_m['F1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds, probs = model.predict(adj, features, timestamps, adj_per_type)
    y_true = labels[test_mask].cpu().numpy()
    y_pred = preds[test_mask].cpu().numpy()
    y_prob = probs[test_mask, 1].cpu().numpy()
    test_m = calculate_metrics(y_true, y_pred, y_prob)
    print_metrics(test_m, "Test")

    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    result = {
        'model': 'TP-THGN-GPU',
        'dataset': args.dataset,
        'seed': args.seed,
        'config': config,
        'test_metrics': {k: v for k, v in test_m.items() if k != 'Confusion_Matrix'},
        'training_log': training_log,
        'training_time_seconds': total_time,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A',
        'use_amp': use_amp,
    }
    out_path = f'experiments/results/tp_thgn_gpu_{args.dataset}_seed{args.seed}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(best_state, f'checkpoints/tp_thgn_gpu_{args.dataset}_best.pth')

    # GPU memory summary
    if device.type == 'cuda':
        print(f"\nGPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB peak")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TP-THGN (GPU version)')
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
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    args = parser.parse_args()
    train(args)
