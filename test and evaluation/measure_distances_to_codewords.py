import os
import sys
import json
# Get the absolute path of the directory containing the current script.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go two levels up to reach the "hazanom" folder.
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Print for debugging (optional)
print("Current directory:", current_dir)
print("parent directory:", parent_dir)

# Add the grandparent directory to sys.path if it's not already there.
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# your imports
from models import VQVAE_s_hcf
from liverpool.closeset.dataset_preparation import LoadDataset, awgn, ChannelIndSpectrogram
from hand_crafted_features import compute_rf_features

def load_full_dataset(config):
    # 1. Load IQ samples and labels.
    loader = LoadDataset()
    data, labels = loader.load_iq_samples(
        file_path=config['path_to_lora_train'],
        dev_range=config['dev_range'],
        pkt_range=config['pkt_range']
    )
    # 2. Adjust labels
    labels = labels - config['dev_range'][0]
    labels = torch.tensor(labels, dtype=torch.long)
    # 3. AWGN
    # data = awgn(data, config['snr_range'])
    # 4. To spectrograms
    spec = ChannelIndSpectrogram().channel_ind_spectrogram(data)
    spec = torch.tensor(spec, dtype=torch.float32).permute(0,3,1,2)
    # 5. Extra features
    extra = []
    for sig in data:
        amp_imb, _, cfo, lo = compute_rf_features(sig)
        extra.append([amp_imb, cfo, lo])
    extra = torch.tensor(np.array(extra), dtype=torch.float32)
    # 6. Single TensorDataset
    ds = TensorDataset(spec, extra, labels)
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)

def compute_and_save_distances(config, model_ckpt, out_dir):
    # 1. Load config & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # 2. DataLoader
    loader = load_full_dataset(config)

    # 3. Prepare storage
    all_dists = []
    all_labels = []

    with torch.no_grad():
        for x, extra, y in loader:
            x, extra = x.to(device), extra.to(device)
            # encode + concat
            z_e = model.encoder(x)                              # [B, num_embeddings]
            extra_out = model.extra_fc(extra)                   # [B, hcf_n]
            combined = model.concat_fc(torch.cat([z_e, extra_out], dim=1))
            # get codebook embeddings
            # adjust attribute name if necessary:
            # e.g. model.quantizer.embedding.weight or model.quantizer.codebook
            embeddings = model.quantizer.codebook.weight         # [128, num_embeddings]
            # compute distances [B, 128]
            d = torch.sum((combined.unsqueeze(1) - embeddings.unsqueeze(0))**2, dim=-1)
            all_dists.append(d.cpu())
            all_labels.append(y)

    all_dists = torch.cat(all_dists, dim=0).numpy()    # shape (30000,128)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # shape (30000,)

    os.makedirs(out_dir, exist_ok=True)
    # 4. Save full arrays
    np.save(os.path.join(out_dir, 'distances_all.npy'), all_dists)
    np.save(os.path.join(out_dir, 'labels_all.npy'), all_labels)

    # 5. Save one file per class
    for c in range(config['num_classes']):
        idx = np.where(all_labels == c)[0]
        class_dists = all_dists[idx]   # shape (1000,128)
        np.save(os.path.join(out_dir, f'class_{c:02d}_dists.npy'), class_dists)

    embeddings = model.quantizer.codebook.weight.data  # [128, D]
    logits     = model.classifier(embeddings)          # [128, num_classes]
    preds      = torch.argmax(logits, dim=1).cpu().numpy()  # [128]

    code_to_class = {
        f"codeword {i}": int(preds[i])
        for i in range(len(preds))
    }
    with open(os.path.join(out_dir, 'code_to_class.json'), 'w') as f:
        json.dump(code_to_class, f, indent=2)

    print(f"Saved distances + per-class files + code_to_class.json in '{out_dir}'")

if __name__ == "__main__":
    # --- EDIT THESE TO YOUR SETTINGS ---
    config = {
        'path_to_lora_train': 'LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5',
        'dev_range': np.arange(30,40, dtype=int),
        'pkt_range': np.arange(0,100, dtype=int),
        'snr_range': np.arange(20,80),
        'batch_size': 256,
        'use_hcf': True,
        'input_shape': (1,102,62),     # adjust as needed
        'num_embeddings': 128,          # your latent dim
        'hcf_n': 30,
        'codebook_size': 128,
        'num_classes': 10,
        'commitment_loss_weight': 0.1
    }
    model_ckpt = 'vq_vae_models/five_stage_models/hcf/stage_5/best_classifier_hcf_dev_30_40_100.pth'
    out_dir = 'analysis/distances_liv'
    compute_and_save_distances(config, model_ckpt, out_dir)
