"""Visualization utilities for paper figures."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_training_curve(log_data, output_path='figures/training_curve.pdf'):
    """Fig.2: Training convergence curve."""
    epochs = log_data['epochs']
    losses = log_data['losses']
    val_f1s = log_data['val_f1s']

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


def plot_attention_heatmap(gate_weights, relation_weights,
                           output_path='figures/attention_heatmap.pdf'):
    """Fig.4: Attention weight heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    gw = np.array(gate_weights).reshape(1, -1)
    sns.heatmap(gw, ax=axes[0], annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'Head {i+1}' for i in range(len(gate_weights))],
                yticklabels=['Weight'])
    axes[0].set_title('Information Perception Gate Weights')

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
