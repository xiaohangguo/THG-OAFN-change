"""
Update comparison results with TP-THGN v3 data and regenerate paper figures.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np

# Load v3 multiseed results
with open('experiments/results/tp_thgn_v3_multiseed.json') as f:
    v3_data = json.load(f)

# Load existing comparison results
with open('experiments/results/comparison_results.json') as f:
    comparison = json.load(f)

# Replace the old TP-THGN entry with v3 results
v3_entry = {
    'model': 'TP-THGN (ours)',
    'variant': 'TP-THGN v3 (Feature-Dominant + Gated Graph)',
    'AUC': v3_data['summary']['AUC'],
    'F1': v3_data['summary']['F1'],
    'Recall': v3_data['summary']['Recall'],
    'Precision': v3_data['summary']['Precision'],
    'AUPRC': v3_data['summary']['AUPRC'],
    'G-Mean': v3_data['summary']['G-Mean'],
}

# Find and replace old TP-THGN entry
updated = []
for entry in comparison:
    if entry.get('model') == 'TP-THGN (ours)':
        updated.append(v3_entry)
    else:
        updated.append(entry)

# If TP-THGN wasn't found, append
if not any(e.get('model') == 'TP-THGN (ours)' for e in updated):
    updated.append(v3_entry)

with open('experiments/results/comparison_results.json', 'w') as f:
    json.dump(updated, f, indent=2)
print("Updated comparison_results.json with TP-THGN v3 data")

# Also update ablation_results.json with v3 ablation data
with open('experiments/results/ablation_v3_results.json') as f:
    ablation_v3 = json.load(f)

# Convert to standard format
ablation_formatted = []
for entry in ablation_v3:
    ablation_formatted.append({
        'variant': entry['model'],
        'model': entry['model'],
        **{k: v for k, v in entry['summary'].items()},
    })

with open('experiments/results/ablation_results.json', 'w') as f:
    json.dump(ablation_formatted, f, indent=2)
print("Updated ablation_results.json with v3 ablation data")

# Now regenerate figures
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: Comparison bar chart
    models = [e['model'] for e in updated]
    f1_means = [e['F1']['mean'] for e in updated]
    f1_stds = [e['F1']['std'] for e in updated]
    auc_means = [e['AUC']['mean'] for e in updated]

    # Short names for x-axis
    short_names = {
        'Logistic Regression': 'LR',
        'XGBoost': 'XGBoost',
        'GCN': 'GCN',
        'GAT': 'GAT',
        'GraphSAGE': 'GraphSAGE',
        'TP-THGN (ours)': 'TP-THGN\n(ours)',
    }

    x_labels = [short_names.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, f1_means, width, yerr=f1_stds, label='F1-Score',
                   color='#2196F3', alpha=0.85, capsize=3)
    bars2 = ax.bar(x + width/2, auc_means, width, label='AUC-ROC',
                   color='#FF9800', alpha=0.85)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison on Amazon Dataset', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.88, color='red', linestyle='--', alpha=0.5, label='F1 target (0.88)')

    for bar, val in zip(bars1, f1_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/comparison_bar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/comparison_bar.pdf/.png")

    # Figure 2: Ablation bar chart
    ablation_names = [e['model'] for e in ablation_v3]
    ablation_f1 = [e['summary']['F1']['mean'] for e in ablation_v3]
    ablation_f1_std = [e['summary']['F1']['std'] for e in ablation_v3]
    ablation_auc = [e['summary']['AUC']['mean'] for e in ablation_v3]

    short_ablation = ['Full', 'w/o Graph', 'w/o SMOTE', 'w/o Focal', 'w/o Gate']

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ablation_names))
    colors = ['#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#2196F3']

    bars = ax.bar(x, ablation_f1, 0.6, yerr=ablation_f1_std, color=colors,
                  alpha=0.85, capsize=4)
    ax.set_xlabel('Model Variant', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Ablation Study — Component Contributions', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_ablation, fontsize=10)
    ax.set_ylim(0.82, 0.94)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, ablation_f1):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/ablation_bar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/ablation_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/ablation_bar.pdf/.png")

    # Figure 3: Multi-metric radar/comparison for paper
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ['F1', 'AUC', 'Recall', 'Precision', 'AUPRC', 'G-Mean']
    highlight_models = ['XGBoost', 'GraphSAGE', 'TP-THGN (ours)']
    colors_h = ['#FF9800', '#2196F3', '#4CAF50']
    x_m = np.arange(len(metrics))

    for model_name, color in zip(highlight_models, colors_h):
        entry = next(e for e in updated if e['model'] == model_name)
        values = [entry[m]['mean'] for m in metrics]
        ax.plot(x_m, values, 'o-', color=color, linewidth=2, markersize=8, label=model_name)

    ax.set_xticks(x_m)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Multi-Metric Comparison: Top Models', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/multi_metric_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/multi_metric_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/multi_metric_comparison.pdf/.png")

except ImportError as e:
    print(f"matplotlib not available: {e}")

print("\nDone! All figures and results updated.")
