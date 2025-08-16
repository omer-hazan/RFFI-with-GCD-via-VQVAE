import os
import json
import math

# Load the aggregated metrics summary
def load_summary(path):
    with open(path, 'r') as f:
        return json.load(f)

# Compute combined stats (equal-weight average of correct and incorrect)
def combine_stats(correct_stats, incorrect_stats):
    combined = {}
    for metric, stats_c in correct_stats.items():
        mean_c, var_c = stats_c['mean'], stats_c['var']
        mean_i, var_i = incorrect_stats[metric]['mean'], incorrect_stats[metric]['var']
        mean_s = 0.5 * (mean_c + mean_i)
        var_s = 0.5 * (var_c + var_i)
        combined[metric] = {'mean': mean_s, 'var': var_s}
    return combined

# Compute Cohen's d for each metric between two groups
def compute_cohens_d(group1, group2):
    d = {}
    for metric, stats1 in group1.items():
        mean1, var1 = stats1['mean'], stats1['var']
        mean2, var2 = group2[metric]['mean'], group2[metric]['var']
        pooled_sd = math.sqrt(0.5 * (var1 + var2))
        d[metric] = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else float('nan')
    return d

# Entry point
def main():
    summary_path = os.path.join('analysis', 'metrics_summary.json')
    summary = load_summary(summary_path)

    # Extract stats
    unseen_seen_devices_corr = summary['unseen_30_39_correct']
    unseen_seen_devices_inc = summary['unseen_30_39_incorrect']
    unseen_devices = summary['unseen_40_44']

    # Combine stats
    unseen_seen_combined = combine_stats(unseen_seen_devices_corr, unseen_seen_devices_inc)

    # Compute effect sizes
    d_unseen_seen_vs_unseen = compute_cohens_d(unseen_seen_combined, unseen_devices)

    # Rank metrics by absolute effect size
    ranked = sorted(d_unseen_seen_vs_unseen.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Metric\tCohen's d (unseen samples from seen devices vs. unseen devices)")
    for metric, d_val in ranked:
        print(f"{metric:20s} {d_val: .4f}")

    best_metric, best_d = ranked[0]
    print(f"\n>>> Best metric to distinguish unseen samples from seen devices vs unseen devices: '{best_metric}' (|d|={abs(best_d):.4f})")

if __name__ == '__main__':
    main()
