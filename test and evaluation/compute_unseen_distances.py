import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --- adjust this so Python can find your project root ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# your imports
from models import VQVAE_s_hcf
from liverpool.closeset.dataset_preparation import LoadDataset, ChannelIndSpectrogram
from hand_crafted_features import compute_rf_features

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
    data  = np.concatenate([data_leg, data_rog], axis=0)
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
                      num_workers=config.get('num_workers',4))

def compute_and_save_unseen_distances(config, model_ckpt, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. load model
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

    # 2. load test (unseen) data
    loader = load_unseen_dataset(config)

    all_dists  = []
    all_labels = []

    # 3. infer + distances
    with torch.no_grad():
        for x, extra, y in loader:
            x, extra = x.to(device), extra.to(device)
            z_e       = model.encoder(x)
            extra_out = model.extra_fc(extra)
            combined  = model.concat_fc(torch.cat([z_e, extra_out], dim=1))

            # codebook embeddings
            emb = model.quantizer.codebook.weight   # [128, D]
            # squared‐Euclid distances: [B,128]
            d = torch.sum((combined.unsqueeze(1) - emb.unsqueeze(0))**2, dim=-1)

            all_dists.append(d.cpu())
            all_labels.append(y)

    all_dists  = torch.cat(all_dists, 0).numpy()   # (N_unseen,128)
    all_labels = torch.cat(all_labels,0).numpy()   # (N_unseen,)

    # 4. save
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'unseen_distances.npy'), all_dists)
    np.save(os.path.join(out_dir, 'unseen_labels.npy'), all_labels)

    # 5. per‐device dumps
    unseen_devices = list(config['dev_range_legitimate']) + list(config['dev_range_rogue'])
    for dev in unseen_devices:
        idxs = np.where(all_labels == dev)[0]
        np.save(os.path.join(out_dir, f'class_{dev:02d}_unseen.npy'),
                all_dists[idxs])

    print(f"Saved unseen distances for devices {unseen_devices} in '{out_dir}'")

if __name__ == "__main__":
    config = {
        # legitimate test data (devices 30–39, pkts 100–199)
        'file_path_legitimate': 'LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5',
        'dev_range_legitimate': np.arange(30, 40, dtype=int),
        'pkt_range_legitimate': np.arange(100, 200, dtype=int),

        # rogue test data (devices 40–44, pkts 0–99)
        'file_path_rogue':       'LoRa_RFFI_dataset/dataset/Test/dataset_rogue.h5',
        'dev_range_rogue':       np.arange(40, 45, dtype=int),
        'pkt_range_rogue':       np.arange(0, 100, dtype=int),

        # common dataloader settings
        'batch_size': 256,
        'input_shape': (1, 102, 62),
        'num_embeddings': 128,
        'hcf_n': 30,
        'codebook_size': 128,
        # NOTE: num_classes is still your original training classes (0–29)
        'num_classes': 10,
        'commitment_loss_weight': 0.1,
    }

    model_ckpt = 'vq_vae_models/five_stage_models/hcf/stage_5/best_classifier_hcf_dev_30_40_100.pth'
    out_dir     = 'analysis/unseen_distances_liv'

    compute_and_save_unseen_distances(config, model_ckpt, out_dir)
