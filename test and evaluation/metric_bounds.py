import os
import json
import numpy as np
import argparse

def analyze_percentiles(metric_name, step, summary_path):
    # Load summary JSON
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Extract mean and variance for correct predictions in devices 30–39
    group = summary['unseen_30_39_correct']
    if metric_name not in group:
        raise ValueError(f"Metric '{metric_name}' not found in summary file.")

    mu = group[metric_name]['mean']
    std = np.sqrt(group[metric_name]['var'])

    # Load labels to filter devices 30–39
    base_dir = os.path.dirname(summary_path)
    # labels_u_np = np.load(os.path.join(base_dir, 'unseen_distances_liv', 'unseen_labels.npy')).astype(int).ravel()
    # mask_30_39 = (labels_u_np >= 30) & (labels_u_np < 40)

    # Load metric values
    metrics_file = os.path.join(base_dir, 'metrics_seen_unseen.npy')
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Expected file: {metrics_file}")
    metrics = np.load(metrics_file, allow_pickle=True).item()

    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' not found in file {metrics_file}")

    values = metrics[metric_name]
    
    # Print header
    print(f"\nMetric: '{metric_name}' | Mean: {mu:.4f}, Std: {std:.4f}")
    print(f"{'n':>4} | {'% in range':>12} | {'Lower Bound':>12} | {'Upper Bound':>12}")
    print("-" * 48)

    # Iterate over n from 1 to 3 with custom step
    n_values = np.arange(1.0, 3.0 + step, step)
    for n in n_values:
        lower = mu - n * std
        upper = mu + n * std
        in_bounds = (values >= lower) & (values <= upper)
        percentage = 100.0 * np.sum(in_bounds) / len(values)
        print(f"{n:4.1f} | {percentage:12.2f}% | {lower:12.4f} | {upper:12.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, required=True, help="Metric name (e.g., entropy, highest_peak)")
    parser.add_argument('--step', type=float, default=0.5, help="Step size for n (from 1 to 3)")
    parser.add_argument('--summary', type=str, default='analysis/metrics_summary.json', help="Path to summary JSON file")
    args = parser.parse_args()

    analyze_percentiles(args.metric, args.step, args.summary)
