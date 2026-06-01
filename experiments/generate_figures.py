"""Generate all paper figures from experiment results."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import (
    plot_ablation_bar,
    plot_comparison_table,
    plot_training_curve,
    plot_attention_heatmap,
)


def main():
    os.makedirs('figures', exist_ok=True)

    # Fig.3: Ablation bar chart
    if os.path.exists('experiments/results/ablation_results.json'):
        print("Generating ablation bar chart...")
        plot_ablation_bar('experiments/results/ablation_results.json')
    else:
        print("SKIP: ablation_results.json not found")

    # Tab.3: Comparison table
    if os.path.exists('experiments/results/comparison_results.json'):
        print("Generating comparison table...")
        plot_comparison_table('experiments/results/comparison_results.json')
    else:
        print("SKIP: comparison_results.json not found")

    # Fig.2: Training curve (from any TP-THGN result with training_log)
    for f in os.listdir('experiments/results'):
        if f.startswith('tp_thgn_') and f.endswith('.json'):
            with open(os.path.join('experiments/results', f)) as fh:
                data = json.load(fh)
            if 'training_log' in data and data['training_log']['epochs']:
                print(f"Generating training curve from {f}...")
                plot_training_curve(data['training_log'])
                break

    print("\nAll available figures generated in figures/")


if __name__ == '__main__':
    main()
