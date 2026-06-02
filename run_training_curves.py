"""
Generate training convergence curves for TP-THGN v3.
Records loss, F1, AUC per epoch and plots Fig.2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import numpy as np
import torch.nn.functional as F
from models.tp_thgn_gpu import create_tp_thgn_gpu_model
from utils.data_loader import load_real_dataset, split_data
from utils.graph_utils import build_sparse_adj
from utils.metrics import calculate_metrics

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

config = {'input_dim': 25, 'hidden_dim': 128, 'output_dim': 2, 'num_relations': 3,
          'dropout': 0.3, 'oversample_ratio': 1.5, 'beta_laplacian': 0.01, 'focal_gamma': 2.0}

model = create_tp_thgn_gpu_model(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

history = {'epoch': [], 'train_loss': [], 'val_f1': [], 'val_auc': [], 'val_recall': [], 'val_precision': [], 'lr': []}

print("Recording training curves (300 epochs, eval every 5)...")
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    logits, _, losses = model(adj, features, labels, None,
                              adj_per_type=adj_per_type, training=True, train_mask=train_mask_t)
    loss = losses['total_loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            preds, probs = model.predict(adj, features, None, adj_per_type)
        y_true = labels[val_mask_t].cpu().numpy()
        y_pred = preds[val_mask_t].cpu().numpy()
        y_prob = probs[val_mask_t, 1].cpu().numpy()
        val_m = calculate_metrics(y_true, y_pred, y_prob)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(loss.item())
        history['val_f1'].append(val_m['F1'])
        history['val_auc'].append(val_m['AUC'])
        history['val_recall'].append(val_m['Recall'])
        history['val_precision'].append(val_m['Precision'])
        history['lr'].append(scheduler.get_last_lr()[0])

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Loss: {loss.item():.4f} | F1: {val_m['F1']:.4f} | AUC: {val_m['AUC']:.4f}")

os.makedirs('experiments/results', exist_ok=True)
with open('experiments/results/training_curves.json', 'w') as f:
    json.dump(history, f, indent=2)
print("Saved training curves data.")

# Generate figure
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(history['epoch'], history['train_loss'], 'b-', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('(a) Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['epoch'], history['val_f1'], 'r-', linewidth=1.5, label='F1')
    axes[1].plot(history['epoch'], history['val_auc'], 'g--', linewidth=1.5, label='AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('(b) Validation F1 & AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    axes[2].plot(history['epoch'], history['val_recall'], 'm-', linewidth=1.5, label='Recall')
    axes[2].plot(history['epoch'], history['val_precision'], 'c-', linewidth=1.5, label='Precision')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].set_title('(c) Recall & Precision')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved figures/training_curves.pdf and .png")
    plt.close()
except ImportError:
    print("matplotlib not available, skipping figure generation")
