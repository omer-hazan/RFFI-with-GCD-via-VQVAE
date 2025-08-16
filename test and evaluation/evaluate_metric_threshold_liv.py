import os
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_distances(base_dir):
    """
    Load distances and labels for:
    - Unseen samples from seen devices (30–39) → label 0
    - Unseen samples from unseen devices (40–44) → label 1
    Returns torch distances tensor on device and labels numpy array.
    """
    dist_u = np.load(os.path.join(base_dir, 'unseen_distances_liv', 'unseen_distances.npy'))
    labels_u = np.load(os.path.join(base_dir, 'unseen_distances_liv', 'unseen_labels.npy')).astype(int).ravel()

    # Filter
    mask_seen = (labels_u >= 30) & (labels_u < 40)
    mask_rogue = (labels_u >= 40) & (labels_u <= 44)

    dist_seen = dist_u[mask_seen]
    dist_rogue = dist_u[mask_rogue]

    # Create binary labels
    labels_seen = np.zeros(dist_seen.shape[0], dtype=int)   # 0 = unseen samples from seen devices
    labels_rogue = np.ones(dist_rogue.shape[0], dtype=int)  # 1 = rogue (unseen devices)

    # Concatenate
    distances = np.concatenate([dist_seen, dist_rogue], axis=0)
    labels = np.concatenate([labels_seen, labels_rogue], axis=0)

    distances_t = torch.from_numpy(distances).float().to(device)
    return distances_t, labels

def compute_metric_torch(distances, metric):
    eps = 1e-8
    inv = 1.0 / (distances + eps)
    if metric == 'entropy':
        p = inv / inv.sum(dim=1, keepdim=True)
        return - (p * torch.log(p + eps)).sum(dim=1)
    elif metric == 'relative_peak':
        top2 = torch.topk(inv, 2, dim=1).values
        return (top2[:,0] - top2[:,1]) / (top2[:,0] + eps)
    elif metric == 'highest_peak':
        return inv.max(dim=1).values
    elif metric == 'mean_inv_distance':
        return inv.mean(dim=1)
    elif metric == 'median_inv_distance':
        return torch.median(inv, dim=1).values
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy of thresholded metric for unseen samples from seen devices (30–39) vs rogue devices (40–44)')
    parser.add_argument('--metric', required=True,
                        choices=['entropy','relative_peak','highest_peak',
                                 'mean_inv_distance','median_inv_distance'],
                        help='Metric to use')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Threshold on metric to classify as class 1')
    parser.add_argument('--direction', choices=['ge','le'], default='ge',
                        help='"ge" → score >= threshold → class 1, else 0; "le" → score <= threshold → class 1')
    parser.add_argument('--base_dir', default='analysis',
                        help='Base dir with unseen_distances_liv')
    args = parser.parse_args()

    distances_t, labels = load_distances(args.base_dir)

    scores_t = compute_metric_torch(distances_t, args.metric)
    scores = scores_t.cpu().numpy()

    if args.direction == 'ge':
        y_pred = (scores >= args.threshold).astype(int)
    else:
        y_pred = (scores <= args.threshold).astype(int)

    acc = accuracy_score(labels, y_pred)
    cm = confusion_matrix(labels, y_pred)
    report = classification_report(labels, y_pred,
                                   target_names=['unseen_seen', 'unseen_rogue'], digits=4)

    print(f"Metric: {args.metric}, Threshold: {args.threshold}, Direction: {args.direction}")
    print(f"# Unseen from seen devices (label 0): {(labels==0).sum()}")
    print(f"# Unseen (rogue) devices (label 1):    {(labels==1).sum()}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

if __name__ == '__main__':
    main()
