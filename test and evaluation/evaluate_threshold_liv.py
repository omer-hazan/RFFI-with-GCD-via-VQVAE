import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(base_dir):
    """
    Loads distances and labels for unseen samples from seen devices (30-39)
    and completely unseen devices (40-44).
    Returns torch tensors on `device` and binary labels numpy array.
    """
    # Load unseen distances and labels
    dist_u = np.load(os.path.join(base_dir, 'unseen_distances_liv', 'unseen_distances.npy'))
    labels_u = np.load(os.path.join(base_dir, 'unseen_distances_liv', 'unseen_labels.npy')).astype(int).ravel()

    # Filter devices [30-39] (seen) and [40-44] (unseen)
    mask_seen = (labels_u >= 30) & (labels_u < 40)
    mask_unseen = (labels_u >= 40) & (labels_u <= 44)

    dist_seen = dist_u[mask_seen]
    dist_unseen = dist_u[mask_unseen]

    # Create binary labels: 0 = unseen samples from seen devices, 1 = completely unseen devices
    y_seen = np.zeros(len(dist_seen), dtype=int)
    y_unseen = np.ones(len(dist_unseen), dtype=int)

    # Concatenate distances and labels
    dist_t = torch.from_numpy(np.vstack((dist_seen, dist_unseen))).float().to(device)
    y_true = np.concatenate((y_seen, y_unseen))

    return dist_t, y_true


def compute_metric_torch(distances, metric):
    """
    Compute per-sample metric on 1/distances using torch.
    """
    eps = 1e-8
    inv = 1.0 / (distances + eps)
    if metric == 'entropy':
        p = inv / torch.sum(inv, dim=1, keepdim=True)
        return -torch.sum(p * torch.log(p + eps), dim=1)
    elif metric == 'relative_peak':
        top2 = torch.topk(inv, 2, dim=1).values
        top1, top2v = top2[:, 0], top2[:, 1]
        return (top1 - top2v) / (top1 + eps)
    elif metric == 'highest_peak':
        return torch.max(inv, dim=1).values
    elif metric == 'mean_inv_distance':
        return torch.mean(inv, dim=1)
    elif metric == 'median_inv_distance':
        return torch.median(inv, dim=1).values
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ROC, AUC, and EER between unseen samples from seen devices (30-39) and completely unseen devices (40-44)')
    parser.add_argument('--metric', required=True,
                        choices=['entropy', 'relative_peak', 'highest_peak',
                                 'mean_inv_distance', 'median_inv_distance'],
                        help='Metric to evaluate')
    parser.add_argument('--invert', action='store_true',
                        help='Invert the metric scores (useful if ROC is below diagonal)')
    parser.add_argument('--base_dir', default='analysis',
                        help='Base directory with unseen_distances subfolder')
    args = parser.parse_args()
    output_dir = f'analysis/ROC/{args.metric}_unseen_vs_unseen_devices_ROC.png'
    print(f"Using device: {device}")

    # Load data
    distances_t, y_true = load_data(args.base_dir)

    # Compute metric
    scores_t = compute_metric_torch(distances_t, args.metric)
    if args.invert:
        scores_t = -scores_t
    scores = scores_t.cpu().numpy()

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Compute EER
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    thr_eer = thresholds[eer_idx]

    print(f"Metric: {args.metric}{' (inverted)' if args.invert else ''}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f} at threshold {thr_eer:.4f}")

    # Plot ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}\nEER = {eer:.4f}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({args.metric})')
    plt.legend(loc='lower right')
    plt.grid(True)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    plt.savefig(output_dir, dpi=150)
    print(f"ROC plot saved to {output_dir}")


if __name__ == '__main__':
    main()