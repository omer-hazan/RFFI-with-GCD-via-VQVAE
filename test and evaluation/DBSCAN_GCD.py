import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, TensorDataset

# ------------------- Setup Paths -------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ------------------- Imports -------------------
from models import VQVAE_s_hcf
from liverpool.closeset.dataset_preparation import LoadDataset, ChannelIndSpectrogram
from hand_crafted_features import compute_rf_features

# ------------------- Config -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256

config = {
    'input_shape': (1, 102, 62),
    'num_embeddings': 128,
    'hcf_n': 30,
    'codebook_size': 128,
    'num_classes': 10,
    'commitment_loss_weight': 0.1,
}
model_ckpt = 'vq_vae_models/five_stage_models/hcf/stage_5/infer_100_no_noise.pth'

# ------------------- Load Model -------------------
model = VQVAE_s_hcf(**config).to(device)
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model.eval()

# ------------------- Load Dataset -------------------
def load_dataset(file_path, dev_range, pkt_range):
    loader = LoadDataset()
    data, labels = loader.load_iq_samples(file_path, dev_range, pkt_range)
    labels = torch.tensor(labels, dtype=torch.long)
    spec = ChannelIndSpectrogram().channel_ind_spectrogram(data)
    spec = torch.tensor(spec, dtype=torch.float32).permute(0, 3, 1, 2)
    extra_feats = []
    for sig in data:
        amp, _, cfo, lo = compute_rf_features(sig)
        extra_feats.append([amp, cfo, lo])
    extra_feats = torch.tensor(np.array(extra_feats), dtype=torch.float32)
    ds = TensorDataset(spec, extra_feats, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

loader_leg = load_dataset('LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5',
                          np.arange(30, 40, dtype=int),
                          np.arange(100, 200, dtype=int))
loader_rog = load_dataset('LoRa_RFFI_dataset/dataset/Test/dataset_rogue.h5',
                          np.arange(40, 45, dtype=int),
                          np.arange(0, 100, dtype=int))

loader_combined = list(loader_leg) + list(loader_rog)

# ------------------- Extract Latents -------------------
def extract_latents(dataloader):
    latents = []
    labels = []
    with torch.no_grad():
        for x, extra, y in dataloader:
            x, extra = x.to(device), extra.to(device)
            _, _, _, combined = model(x, extra, return_latents=True)
            latents.append(combined.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)

latents, labels = extract_latents(loader_combined)
labels = labels.ravel()
# ------------------- DBSCAN on Rogue -------------------
rogue_mask = labels >= 40
rogue_latents = latents[rogue_mask]
rogue_labels = labels[rogue_mask]

dbscan = DBSCAN(eps=0.372648, min_samples=50)
cluster_labels = dbscan.fit_predict(rogue_latents)

# ------------------- Evaluation -------------------
n_true_classes = len(np.unique(rogue_labels))
n_clusters_found = len(np.unique(cluster_labels[cluster_labels != -1]))

print(f"Expected rogue classes: {n_true_classes}")
print(f"Discovered DBSCAN clusters (excluding noise): {n_clusters_found}")

if n_clusters_found > 1:
    sil = silhouette_score(rogue_latents[cluster_labels != -1], cluster_labels[cluster_labels != -1])
else:
    sil = float('nan')
print(f"Silhouette Score: {sil:.3f}")

# Cluster purity report
mask = cluster_labels != -1
valid_clusters = cluster_labels[mask]
valid_labels = rogue_labels[mask]

cluster_to_class_map = defaultdict(list)
for cluster_id, true_label in zip(valid_clusters, valid_labels):
    cluster_to_class_map[cluster_id].append(true_label)

print("\nCluster Purity Report:")
total_correct = 0
total_samples = 0
for cluster_id, label_list in cluster_to_class_map.items():
    count = Counter(label_list)
    dominant_class, dominant_count = count.most_common(1)[0]
    cluster_size = len(label_list)
    purity = dominant_count / cluster_size
    total_correct += dominant_count
    total_samples += cluster_size
    print(f"  Cluster {cluster_id}: Size={cluster_size}, Purity={purity:.2f}, Dominant Class={dominant_class}")

overall_purity = total_correct / total_samples if total_samples > 0 else 0
print(f"\nOverall weighted purity: {overall_purity:.3f}")

# ------------------- Visualization -------------------
pca = PCA(n_components=2)
proj = pca.fit_transform(rogue_latents)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster_labels, cmap='tab10', s=10, alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.title(f'DBSCAN Clustering | Clusters: {n_clusters_found} | Purity: {overall_purity:.2f}')
plt.tight_layout()
os.makedirs("analysis", exist_ok=True)
plt.savefig("analysis/dbscan_rogue_clusters.png", dpi=60)
plt.close()

# ------------------- Save Outputs -------------------
np.savez("analysis/discovered_clusters_dbscan.npz",
         rogue_latents=rogue_latents,
         true_labels=rogue_labels,
         cluster_labels=cluster_labels)

print("Saved: discovered_clusters_dbscan.npz and dbscan_rogue_clusters.png")
