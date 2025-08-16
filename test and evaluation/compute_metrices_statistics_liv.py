import os
import json
import numpy as np
import torch

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
base_dir   = 'analysis'
dist_dir   = os.path.join(base_dir, 'distances_liv')
unseen_dir = os.path.join(base_dir, 'unseen_distances_liv')

# Load seen data distances and labels
dists_np = np.load(os.path.join(dist_dir, 'distances_all.npy'))  # shape (30000,128)
labels_np = np.load(os.path.join(dist_dir, 'labels_all.npy')).astype(int)  # shape (30000,)
with open(os.path.join(dist_dir, 'code_to_class.json'), 'r') as f:
    code_to_class = {int(k.split()[1]): int(v) for k, v in json.load(f).items()}

# Move to torch tensors on GPU
dists     = torch.from_numpy(dists_np).float().to(device)  # [N_seen, K]
labels    = torch.from_numpy(labels_np).long().to(device)  # [N_seen]

# Predicted classes via nearest codeword
nearest_codes = torch.argmin(dists, dim=1)  # [N_seen]
mapping_keys   = sorted(code_to_class.keys())  # [0..127]
mapping_list   = [code_to_class[k] for k in mapping_keys]
mapping_tensor = torch.tensor(mapping_list, device=device)  # [K]
preds          = mapping_tensor[nearest_codes]               # [N_seen]

# Load unseen data
labels_u_np = np.load(os.path.join(unseen_dir, 'unseen_labels.npy')).astype(int).ravel()
dists_u_np  = np.load(os.path.join(unseen_dir, 'unseen_distances.npy'))

# Separate devices 30-39 (seen) and 40-44 (unseen)
mask_seen_u    = (labels_u_np >= 30) & (labels_u_np < 40)
mask_unseen_u  = (labels_u_np >= 40) & (labels_u_np <= 44)

# Seen devices 30-39
labels_seen_u_np = labels_u_np[mask_seen_u]
dists_seen_u_np  = dists_u_np[mask_seen_u, :]
dists_seen_u = torch.from_numpy(dists_seen_u_np).float().to(device)
labels_seen_u = torch.from_numpy(labels_seen_u_np).long().to(device)

nearest_codes_seen_u = torch.argmin(dists_seen_u, dim=1)
preds_seen_u = mapping_tensor[nearest_codes_seen_u]
true_classes_seen_u = labels_seen_u - 30
classified_correctly_seen_u = (preds_seen_u == true_classes_seen_u)
accuracy_seen_u = classified_correctly_seen_u.float().mean().item()

# Unseen devices 40-44
labels_unseen_np = labels_u_np[mask_unseen_u]
dists_unseen_np  = dists_u_np[mask_unseen_u, :]
dists_unseen = torch.from_numpy(dists_unseen_np).float().to(device)
labels_unseen = torch.from_numpy(labels_unseen_np).long().to(device)

# Metric computation using GPU
def compute_sample_metrics_torch(dists_tensor):
    eps = 1e-8
    inv = 1.0 / (dists_tensor + eps)
    p = inv / torch.sum(inv, dim=1, keepdim=True)
    entropy = -torch.sum(p * torch.log(p + eps), dim=1)
    top2_vals = torch.topk(inv, 2, dim=1).values
    top1, top2v = top2_vals[:, 0], top2_vals[:, 1]
    relative_peak = (top1 - top2v) / (top1 + eps)
    highest_peak = top1
    mean_inv    = torch.mean(inv, dim=1)
    median_inv  = torch.median(inv, dim=1).values
    return {
        'entropy': entropy,
        'relative_peak': relative_peak,
        'highest_peak': highest_peak,
        'mean_inv_distance': mean_inv,
        'median_inv_distance': median_inv
    }

metrics_seen_t         = compute_sample_metrics_torch(dists)
metrics_seen_unseen_t  = compute_sample_metrics_torch(dists_seen_u)
metrics_unseen_t       = compute_sample_metrics_torch(dists_unseen)

# Move metrics back to CPU numpy
metrics_seen         = {k: v.cpu().numpy() for k, v in metrics_seen_t.items()}
metrics_seen_unseen  = {k: v.cpu().numpy() for k, v in metrics_seen_unseen_t.items()}
metrics_unseen       = {k: v.cpu().numpy() for k, v in metrics_unseen_t.items()}
np.save(os.path.join(base_dir, 'metrics_seen_unseen.npy'), metrics_seen_unseen)

# Aggregation helper
def aggregate_group_stats(metrics, indices):
    stats = {}
    for name, arr in metrics.items():
        subset = arr[indices]
        stats[name] = {'mean': float(np.mean(subset)), 'var': float(np.var(subset))}
    return stats

# Build summary
summary = {
    'seen_correct': aggregate_group_stats(metrics_seen, np.where(preds.cpu().numpy() == labels_np)[0]),
    'seen_incorrect': aggregate_group_stats(metrics_seen, np.where(preds.cpu().numpy() != labels_np)[0]),
    'unseen_30_39_correct': aggregate_group_stats(metrics_seen_unseen, np.where(classified_correctly_seen_u.cpu().numpy())[0]),
    'unseen_30_39_incorrect': aggregate_group_stats(metrics_seen_unseen, np.where(~classified_correctly_seen_u.cpu().numpy())[0]),
    'accuracy_unseen_30_39': accuracy_seen_u,
    'unseen_40_44': aggregate_group_stats(metrics_unseen, np.arange(len(labels_unseen_np)))
}

# Save summary
out_file = os.path.join(base_dir, 'metrics_summary.json')
with open(out_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Saved aggregated metrics to {out_file} using device {device}")
