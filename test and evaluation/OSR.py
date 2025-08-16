import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
# --- locate project root ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# your modules
from models import VQVAE_s_hcf
from liverpool.closeset.dataset_preparation import LoadDataset, ChannelIndSpectrogram
from hand_crafted_features import compute_rf_features
from create_codebook import lbg_algorithm

def load_unseen_dataset(config):
    loader = LoadDataset()

    # 1) legitimate devices 30–39, packets 100–199
    data_leg, label_leg = loader.load_iq_samples(
        config['file_path_legitimate'],
        config['dev_range_legitimate'],
        config['pkt_range_legitimate']
    )

    # 2) rogue devices 40–44, packets 0–99
    data_rog, label_rog = loader.load_iq_samples(
        config['file_path_rogue'],
        config['dev_range_rogue'],
        config['pkt_range_rogue']
    )

    # 3) concatenate
    data = np.concatenate([data_leg, data_rog], axis=0)
    labels = np.concatenate([label_leg, label_rog], axis=0)

    # 4) spectrograms
    spec = ChannelIndSpectrogram().channel_ind_spectrogram(data)
    spec = torch.tensor(spec, dtype=torch.float32).permute(0, 3, 1, 2)

    # 5) extra HCF features
    extra_feats = []
    for sig in data:
        amp, _, cfo, lo = compute_rf_features(sig)
        extra_feats.append([amp, cfo, lo])
    extra_feats = torch.tensor(np.array(extra_feats), dtype=torch.float32)

    labels = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(spec, extra_feats, labels)
    return DataLoader(ds,
                      batch_size=config['batch_size'],
                      shuffle=False,
                      num_workers=config.get('num_workers', 4))

def evaluate_dbscan(rogue_latents, rogue_labels, save_dir="analysis"):
    os.makedirs(save_dir, exist_ok=True)

    dbscan = DBSCAN(eps=1.3, min_samples=10)
    cluster_labels = dbscan.fit_predict(rogue_latents)

    # --- Evaluation ---
    true_class_ids = np.unique(rogue_labels)
    n_clusters_found = len(np.unique(cluster_labels[cluster_labels != -1]))

    print(f"Expected rogue classes: {len(true_class_ids)} (IDs: {true_class_ids})")    
    print(f"Discovered DBSCAN clusters (excluding noise): {n_clusters_found}")

    sil = silhouette_score(
        rogue_latents[cluster_labels != -1],
        cluster_labels[cluster_labels != -1]
    ) if n_clusters_found > 1 else float('nan')
    print(f"Silhouette Score: {sil:.3f}")

    # --- Cluster Purity ---
    print("\nCluster Purity Report:")
    mask = cluster_labels != -1
    valid_clusters = cluster_labels[mask]
    valid_labels = rogue_labels[mask]

    cluster_to_class_map = defaultdict(list)
    for cluster_id, true_label in zip(valid_clusters, valid_labels):
        cluster_to_class_map[cluster_id].append(int(true_label))

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

    # --- Save Results ---
    np.savez(os.path.join(save_dir, "discovered_clusters_dbscan.npz"),
             rogue_latents=rogue_latents,
             true_labels=rogue_labels,
             cluster_labels=cluster_labels)

    # --- Plot ---
    pca = PCA(n_components=2)
    proj = pca.fit_transform(rogue_latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster_labels, cmap='tab10', s=10, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.title(f'DBSCAN Clustering | Clusters: {n_clusters_found} | Purity: {overall_purity:.2f}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dbscan_rogue_clusters.png"), dpi=300)
    plt.close()

    print("Saved: discovered_clusters_dbscan.npz and dbscan_rogue_clusters.png")
    return cluster_labels

def expand_codebook_with_dbscan_clusters(rogue_latents, cluster_labels, model, codebook_size_per_cluster=12):
    """
    Generate new codewords for each DBSCAN-discovered cluster using LBG algorithm,
    assign labels based on dominant class in cluster, and append to model's codebook.

    Returns:
        new_codewords (Tensor), new_codeword_labels (Tensor)
    """
    if isinstance(rogue_latents, np.ndarray):
        rogue_latents = torch.tensor(rogue_latents, dtype=torch.float32)
    if isinstance(cluster_labels, np.ndarray):
        cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)

    valid_mask = cluster_labels != -1
    if valid_mask.sum() == 0:
        print("No valid DBSCAN clusters found.")
        return None, None

    valid_latents = rogue_latents[valid_mask]
    valid_cluster_ids = cluster_labels[valid_mask]

    new_codewords = []
    new_labels = []

    for cluster_id in torch.unique(valid_cluster_ids):
        cluster_data = valid_latents[valid_cluster_ids == cluster_id]
        if cluster_data.shape[0] < codebook_size_per_cluster:
            print(f"Skipping cluster {cluster_id.item()} (only {cluster_data.shape[0]} samples)")
            continue
        codebook = lbg_algorithm(cluster_data, codebook_size=codebook_size_per_cluster)

        # Infer dominant class label from cluster data
        cluster_labels_np = cluster_data.detach().cpu().numpy()
        cluster_idx = (valid_cluster_ids == cluster_id).nonzero(as_tuple=True)[0]
        cluster_true_labels = model.cluster_true_labels[cluster_idx].cpu().numpy().astype(int).ravel()  
        dominant_label = int(Counter(cluster_true_labels).most_common(1)[0][0])
        label_tensor = torch.full((codebook_size_per_cluster,), dominant_label, dtype=torch.long)

        new_codewords.append(codebook)
        new_labels.append(label_tensor)

    if not new_codewords:
        print("No clusters had enough points to generate codewords.")
        return None, None

    new_codewords = torch.cat(new_codewords, dim=0)
    new_codeword_labels = torch.cat(new_labels, dim=0)

    # --- Append to model codebook ---
    model_device = model.quantizer.codebook.weight.device
    model.quantizer.codebook.weight.data = torch.cat([
        model.quantizer.codebook.weight.data,
        new_codewords.to(model_device)
    ], dim=0)

    print(f"✅ Extended codebook: new size = {model.quantizer.codebook.weight.shape[0]}")
    model_ckpt = 'vq_vae_models/five_stage_models/hcf/stage_5/infer_100_no_noise.pth'
    extended_ckpt_path = model_ckpt.replace(".pth", "_extended.pth")
    torch.save(model.state_dict(), extended_ckpt_path)
    print(f"✅ Model with extended codebook saved to: {extended_ckpt_path}")
    return new_codewords, new_codeword_labels

def classify_with_extended_codebook(config, model, codeword_labels_path):
    """
    Classify new samples based on nearest codeword and its assigned label.
    
    Parameters:
        config (dict): Dataset configuration (paths, batch size, etc.)
        model (nn.Module): VQ-VAE model with extended codebook
        codeword_labels_path (str): Path to saved codeword labels (torch .pt file)

    Returns:
        accuracy (float), y_true (np.ndarray), y_pred (np.ndarray)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # --- Load extended codeword labels ---
    codeword_labels = torch.load(codeword_labels_path)  # shape: [K_total]
    print(codeword_labels)
    assert codeword_labels.shape[0] == model.quantizer.codebook.weight.shape[0], \
        "Mismatch between codebook size and label count"

    # --- Load new unseen test data ---
    loader = LoadDataset()

    data_leg, label_leg = loader.load_iq_samples(
        config['file_path_legitimate'],
        np.arange(30, 40, dtype=int),
        np.arange(200, 300, dtype=int)
    )

    data_rog, label_rog = loader.load_iq_samples(
        config['file_path_rogue'],
        np.arange(40, 42, dtype=int),
        np.arange(100, 200, dtype=int)
    )

    data = np.concatenate([data_leg, data_rog], axis=0)
    labels = np.concatenate([label_leg, label_rog], axis=0)

    # --- Spectrogram + HCF ---
    spec = ChannelIndSpectrogram().channel_ind_spectrogram(data)
    spec = torch.tensor(spec, dtype=torch.float32).permute(0, 3, 1, 2)
    
    extra_feats = []
    for sig in data:
        amp, _, cfo, lo = compute_rf_features(sig)
        extra_feats.append([amp, cfo, lo])
    extra_feats = torch.tensor(extra_feats, dtype=torch.float32)

    y_true = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(spec, extra_feats, y_true)
    test_loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False)

    y_pred = []
    y_true_list = []
    
    with torch.no_grad():
        for x, extra, y in test_loader:
            x, extra = x.to(device), extra.to(device)

            z_e = model.encoder(x)
            extra_out = model.extra_fc(extra)
            z_q = model.concat_fc(torch.cat([z_e, extra_out], dim=1))  # [B, D]

            codebook = model.quantizer.codebook.weight  # [K, D]
            dists = torch.cdist(z_q, codebook)          # [B, K]
            nearest_idx = torch.argmin(dists, dim=1)    # [B]
            predicted = codeword_labels[nearest_idx.cpu()]  # [B]
            y_pred.append(predicted)
            y_true_list.append(y)  # also collect here to be sure

    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true_list).numpy()  # ensures alignment
    y_true = y_true.reshape(-1)
    mask_old = (y_true >= 30) & (y_true < 40)
    acc_old = (y_pred[mask_old] == y_true[mask_old]).mean() if mask_old.any() else float('nan')
    
    # New class: rogue devices (40+)
    mask_new = (y_true >= 40)
    acc_new = (y_pred[mask_new] == y_true[mask_new]).mean() if mask_new.any() else float('nan')
    
    # Overall
    accuracy = (y_pred == y_true).mean()
    
    print(f"\n✅ Classification Summary")
    print(f"Overall Accuracy:      {accuracy:.2%}")
    print(f"Legitimate Accuracy:   {acc_old:.2%}  (Devices 30–39)")
    print(f"Rogue Device Accuracy: {acc_new:.2%}  (Devices ≥ 40)")
    return accuracy, y_true, y_pred
    
def compute_rogue_latents_by_entropy(config, model_ckpt, entropy_threshold, out_latents_path, out_labels_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = VQVAE_s_hcf(
        input_shape=config['input_shape'],
        num_embeddings=config['num_embeddings'],
        hcf_n=config['hcf_n'],
        codebook_size=config['codebook_size'],
        num_classes=config['num_classes'],
        commitment_loss_weight=config['commitment_loss_weight']
    ).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    loader = load_unseen_dataset(config)

    rogue_latents, rogue_labels = [], []
    total_rogue_samples, detected_rogue = 0, 0
    eps = 1e-8
    TP = FP = TN = FN = 0

    with torch.no_grad():
        for x, extra, y in loader:
            x, extra, y = x.to(device), extra.to(device), y.to(device)

            # compute latent vectors
            z_e = model.encoder(x)
            extra_out = model.extra_fc(extra)
            z_q = model.concat_fc(torch.cat([z_e, extra_out], dim=1))

            # distances
            codebook = model.quantizer.codebook.weight
            dists = torch.sum((z_q.unsqueeze(1) - codebook.unsqueeze(0))**2, dim=-1)

            inv_dists = 1.0 / (dists + eps)
            probs = inv_dists / torch.sum(inv_dists, dim=1, keepdim=True)
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)

            mask = (entropy > entropy_threshold)
            is_rogue = (y >= 40).view(-1)
            is_legitimate = ~is_rogue
            TP += (mask & is_rogue).sum().item()      # Detected rogue correctly
            FN += ((~mask) & is_rogue).sum().item()   # Missed rogue
            FP += (mask & is_legitimate).sum().item() # Legitimate misclassified as rogue
            TN += ((~mask) & is_legitimate).sum().item() # Legitimate correctly classified
            rogue_latents.append(z_q[mask.to(z_q.device)].cpu())
            rogue_labels.append(y[mask.to(y.device)].cpu())

            # stats
            total_rogue_samples += (y >= 40).sum().item()
            mask = mask.to(y.device)
            detected_rogue += ((y >= 40) & mask).sum().item()

    rogue_latents = torch.cat(rogue_latents, dim=0).numpy()
    print(f"rogue_latents shape: {rogue_latents.shape}")
    rogue_labels = torch.cat(rogue_labels, dim=0).numpy()

    os.makedirs(os.path.dirname(out_latents_path), exist_ok=True)
    np.save(out_latents_path, rogue_latents)
    np.save(out_labels_path, rogue_labels)

    print(f"Saved {len(rogue_latents)} rogue latents and labels (entropy > {entropy_threshold})")
    print(f"Latents saved to: {out_latents_path}")
    print(f"Labels  saved to: {out_labels_path}")

    # Accuracy of rogue detection
    print(f"\n[Accuracy]")
    print(f"\n[Confusion Matrix]")
    print(f"True Positives (TP): {TP}")
    print(f"False Negatives (FN): {FN}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"Total rogue samples present: {total_rogue_samples}")
    print(f"Rogue samples detected by entropy: {detected_rogue}")
    if total_rogue_samples > 0:
        print(f"Detection Rate: {detected_rogue / total_rogue_samples:.2%}")

    cluster_labels = evaluate_dbscan(rogue_latents, rogue_labels)
    # Assign true labels of rogue_latents to the model temporarily (needed for cluster label mapping)
    model.cluster_true_labels = torch.tensor(rogue_labels, dtype=torch.long)
    
    # Step 1: original codeword labels
    with torch.no_grad():
        original_codewords = model.quantizer.codebook.weight[:config['codebook_size']]
        predicted_logits = model.classifier(original_codewords)
        original_codeword_labels = torch.argmax(predicted_logits, dim=1).cpu() + 30
    
    # Step 2: new codewords + labels
    cluster_labels = evaluate_dbscan(rogue_latents, rogue_labels)
    new_codewords, new_codeword_labels = expand_codebook_with_dbscan_clusters(
        rogue_latents=rogue_latents,
        cluster_labels=cluster_labels,
        model=model,
        codebook_size_per_cluster=16
    )
    
    # Step 3: concatenate all labels
    if new_codewords is not None:
        full_codeword_labels = torch.cat([original_codeword_labels, new_codeword_labels], dim=0)
        torch.save(full_codeword_labels, "analysis/codeword_labels.pt")
        print(f"✅ Saved codeword labels: {full_codeword_labels.shape} → analysis/codeword_labels.pt")
    config['dev_range_legitimate'] = np.arange(30, 40, dtype=int)
    config['pkt_range_legitimate'] = np.arange(200, 300, dtype=int)
    config['dev_range_rogue'] = np.arange(40, 42, dtype=int)
    config['pkt_range_rogue'] = np.arange(100, 200, dtype=int)
    
    # Load model
    model.load_state_dict(torch.load("vq_vae_models/five_stage_models/hcf/stage_5/infer_100_no_noise_extended.pth"))
    
    # Run classification
    acc, y_true, y_pred = classify_with_extended_codebook(
        config,
        model,
        codeword_labels_path="analysis/codeword_labels.pt"
    )



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--entropy_threshold', type=float, required=True, help="Entropy threshold for rogue detection")
        parser.add_argument('--out_latents', type=str, default='analysis/rogue_latents.npy', help="Output .npy file for rogue latents")
        parser.add_argument('--out_labels', type=str, default='analysis/rogue_labels.npy', help="Output .npy file for rogue labels")
        args = parser.parse_args()


    # config settings
        config = {
            'file_path_legitimate': 'LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5',
            'dev_range_legitimate': np.arange(30, 40, dtype=int),
            'pkt_range_legitimate': np.arange(100, 200, dtype=int),
            'file_path_rogue':       'LoRa_RFFI_dataset/dataset/Test/dataset_rogue.h5',
            'dev_range_rogue':       np.arange(40, 42, dtype=int),
            'pkt_range_rogue':       np.arange(0, 100, dtype=int),
            'batch_size': 256,
            'input_shape': (1, 102, 62),
            'num_embeddings': 128,
            'hcf_n': 30,
            'codebook_size': 128,
            'num_classes': 10,
            'commitment_loss_weight': 0.1,
        }

        model_ckpt = 'vq_vae_models/five_stage_models/hcf/stage_5/infer_100_no_noise.pth'
        compute_rogue_latents_by_entropy(
            config,
            model_ckpt,
            args.entropy_threshold,
            out_latents_path=args.out_latents,
            out_labels_path=args.out_labels
        )