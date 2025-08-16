import sys
sys.path.append('/sise/home/hazanom')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import os
from liverpool.closeset.dataset_preparation import LoadDataset, awgn, ChannelIndSpectrogram
from models import VQVAE_no_q, VQVAE_s_no_q, VQVAE_s_no_q_hcf, VQVAE_s_hcf
from liverpool.openset.deep_learning_models_pytorch import TripletNet
#from self_supervised.self_supervised_train import FeatureExtractor
# from self_supervised import dataset_ss 
from hand_crafted_features import compute_rf_features

# --- LBG Quantization Algorithm ---
def lbg_algorithm(data, codebook_size=12, epsilon=0.01, max_iter=100):
    # Initialize codebook with the mean of the data
    codebook = data.mean(dim=0, keepdim=True)

    # Iteratively grow the codebook
    while codebook.size(0) < codebook_size:
        # Split centroids
        perturb = epsilon * torch.randn_like(codebook)
        codebook = torch.cat([codebook + perturb, codebook - perturb], dim=0)

        # Iterative K-means
        for _ in range(max_iter):
            distances = torch.cdist(data, codebook)
            assignments = torch.argmin(distances, dim=1)

            # Update centroids
            new_codebook = torch.zeros_like(codebook)
            counts = torch.zeros(codebook.size(0), device=data.device)

            for i in range(codebook.size(0)):
                assigned_points = data[assignments == i]
                if assigned_points.size(0) > 0:
                    new_codebook[i] = assigned_points.mean(dim=0)
                    counts[i] = assigned_points.size(0)

            if torch.norm(new_codebook - codebook) < 1e-6:
                break
            codebook = new_codebook.clone()

    # Sort by popularity and truncate
    _, top_indices = torch.sort(counts, descending=True)
    codebook = codebook[top_indices[:codebook_size]]  # Keep top centroids

    return codebook


# --- KMeans++ Quantization Algorithm ---
def kmeans_plus_plus(data, codebook_size=30, max_iter=300):
    # Perform KMeans clustering using sklearn
    kmeans = KMeans(n_clusters=codebook_size, init="k-means++", max_iter=max_iter, n_init=10)
    kmeans.fit(data.cpu().numpy())  # Convert to numpy for sklearn

    codebook = torch.tensor(kmeans.cluster_centers_, device=data.device)
    return codebook


# --- Encode and Quantize Function ---
def encode_and_quantize(model, dataloader, device,
                        codebook_size=30, save_path="codebook.pth",
                        method="lbg", hcf=False):
    """
    Extracts latent vectors (with extra features included), then quantizes them
    with either LBG or KMeans++.
    """
    # 1) Extract combined latent vectors (z_e + extra_fc(...) -> concat_fc(...))
    latent_vectors = extract_latent_vectors(model, dataloader, device, hcf)

    # 2) Pick your quantization method
    if method == "lbg":
        codebook = lbg_algorithm(latent_vectors, codebook_size=codebook_size)
    elif method in ["kmeans", "kmeans++"]:
        codebook = kmeans_plus_plus(latent_vectors, codebook_size=codebook_size)
    else:
        raise ValueError("Invalid quantization method. Choose 'lbg' or 'kmeans++'.")

    # 3) Save the resulting codebook
    torch.save(codebook, save_path)
    print(f"Codebook saved to {save_path} using {method.upper()}. Size: {codebook.size(0)}")


def extract_latent_vectors(model, dataloader, device, hcf):
    """
    Runs the inputs + extra_features through the encoder, extra_fc, and concat_fc
    to return the combined latent vector for each sample.
    """
    model.eval()
    latent_vectors = []

    with torch.no_grad():
        for batch in dataloader:
            # batch is (inputs, extra_feats, labels)
            inputs, extra_feats, _ = batch
            inputs = inputs.to(device)
            extra_feats = extra_feats.to(device)

            # z_e is from the FeatureExtractor
            z_e = model.encoder(inputs)  # [B, num_embeddings]
            if hcf:
                _,_,_,combined = model(inputs, extra_feats, return_latents=True)
                latent_vectors.append(combined)
            else:
                latent_vectors.append(z_e)

    # Concatenate all batches into a single tensor
    return torch.cat(latent_vectors, dim=0)

# --- Example Usage ---
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hcf = True
    add_noise = False
    # Load the encoder model
    input_shape = (1, 102, 62)  # Example input shape
    input_shape_ss = (1, 1, 102, 62) 
    model = VQVAE_s_no_q_hcf(input_shape, num_embeddings=128, codebook_size=128, num_classes=30, hcf_n=30)
    # model = TripletNet(input_shape_ss, 0.1)
    # model = FeatureExtractor(input_shape_ss)
    path_to_fe = "vq_vae_models/five_stage_models/hcf/stage_5/best_classifier_hcf_84.53.pth"
    state_dict = torch.load(path_to_fe)

    filtered_state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('norm.') or k.startswith('quantizer.'))}
    model.load_state_dict(filtered_state_dict)
    acc = (path_to_fe.split('.')[0] + '.' + path_to_fe.split('.')[1]).split('_')[-1]
    model.to(device)          # <--- move the entire model to GPU
    model.eval()              # <--- set it to eval mode
    
    LoadDatasetObj = LoadDataset()
    # path_to_lora_train = 'LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5'
    path_to_lora_train = 'LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5'
    dev_range = range(30, 40)
    pkt_range = range(0, 100)
    data_train, label_train = LoadDatasetObj.load_iq_samples(file_path=path_to_lora_train,
                                                             dev_range=dev_range,
                                                             pkt_range=pkt_range)

    # Shuffle data
    index = np.arange(len(label_train))
    np.random.shuffle(index)
    data_train = data_train[index, :]
    label_train = label_train[index]

    # One-hot encoding
    label_train = label_train - dev_range[0]
    label_train = torch.tensor(label_train, dtype=torch.long)

    # Add noise to increase robustness
    if add_noise:
        data_train = awgn(data_train, range(20, 80))

    # Convert to spectrogram
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_train)
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
    dataset = TensorDataset(data, label_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    if hcf:
        extra_features_list = []
        for signal in data_train:
            # compute_rf_features returns (amp imbalance, phase imbalance, CFO, LO leakage)
            amp_imbalance, _, CFO, LO_leakage = compute_rf_features(signal)
            extra_features_list.append([amp_imbalance, CFO, LO_leakage])
        extra_features = np.array(extra_features_list)
        extra_features = torch.tensor(extra_features, dtype=torch.float32)
        dataset = TensorDataset(data, extra_features, label_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    # Choose quantization method: 'lbg' or 'kmeans++'
    method = "lbg"  # Change to 'lbg' if needed
    encode_and_quantize(model, dataloader, device, codebook_size=128,
                        save_path=f"vq_vae_models/five_stage_models/hcf/stage_4_codebook/codebook_dev_30_40_84_53_size_128.pth", method=method, hcf=hcf)
