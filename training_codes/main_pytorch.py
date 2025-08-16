import os
import sys

# Get the absolute path of the directory containing the current script.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go two levels up to reach the "hazanom" folder.
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
# Print for debugging (optional)
print("Current directory:", current_dir)
print("parent directory:", grandparent_dir)

# Add the grandparent directory to sys.path if it's not already there.
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE


from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam

import dataset_preparation
from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram

import deep_learning_models_pytorch
from deep_learning_models_pytorch import TripletNet, identity_loss, TripletNet_hcf, TripletNet_hcf_big
# from playground import feature_representation

from hand_crafted_features import compute_rf_features
from models import VQVAE, VQVAE_s, VQVAE_no_q, VQVAE_s_no_q, VQVAE_s_no_q_hcf, VQVAE_s_hcf, VQVAE_no_q_hcf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import silhouette_score

class RFSingleDataset(Dataset):
    """Return (spectrogram, extra_features, label) per sample."""
    def __init__(self, spectrogram, extra, labels):
        self.spectrogram = spectrogram
        self.extra       = extra
        self.labels      = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.spectrogram[idx],
            self.extra[idx],
            self.labels[idx],
        )

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate false alarm probability, miss detection probability, and accuracy.
    """
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Calculate metrics
    P_FA = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Alarm Probability
    P_MD = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss Detection Probability
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return P_FA, P_MD, accuracy

def train_feature_extractor(
    file_path='LoRa_RFFI_dataset/dataset/Train/dataset_training_aug.h5',
    dev_range=np.arange(0, 30, dtype=int),
    pkt_range=np.arange(0, 1000, dtype=int),
    snr_range=np.arange(20, 80),
    best_model_path='vq_vae_models/triplet_net/',
    pretrained_model_path=None  # <-- NEW ARG
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    train_feature_extractor trains an RFF extractor using triplet loss,
    with an option to load a pretrained model and copy over only its encoder.

    file_path:       path to training dataset
    dev_range:       label range of devices for training
    pkt_range:       packet range from each device for training
    snr_range:       SNR range used for data augmentation
    best_model_path: directory to save the best performing model
    pretrained_model_path: optional path to a PyTorch model file that has
                           .encoder and .classifier. Only .encoder is copied
                           into our TripletNet's encoder.

    Returns:
        Nothing directly, but saves out best_model.pth and prints progress.
    """

    LoadDatasetObj = LoadDataset()

    # 1. Load preamble IQ samples and labels.
    data, label = LoadDatasetObj.load_iq_samples(file_path, dev_range, pkt_range)

    # 2. Add additive Gaussian noise to the IQ samples.
    data = awgn(data, snr_range)

    # 3. Convert time-domain IQ samples to channel-independent spectrograms.
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)

    # 4. Specify hyperparameters.
    margin = 0.1
    batch_size = 32

    # 5. Instantiate TripletNet (this is your feature extractor).
    triplet_net = TripletNet(data.shape, margin).to(device)

    # 6. If a pretrained model path is provided, load only the encoder weights.
    if pretrained_model_path is not None and os.path.isfile(pretrained_model_path):
        print(f"Loading pretrained encoder from: {pretrained_model_path}")
        # Make sure MyPretrainedModel has the same encoder architecture as TripletNet
        pretrained_model = MyPretrainedModel(input_shape=data.shape)
        pretrained_model.load_state_dict(
            torch.load(pretrained_model_path, map_location=device)
        )
        # Copy only the encoder weights
        triplet_net.encoder.load_state_dict(pretrained_model.encoder.state_dict())
        print("Pretrained encoder loaded successfully.")
    else:
        # If path provided but not found, just continue from scratch
        if pretrained_model_path is not None:
            print(f"Warning: {pretrained_model_path} not found. Training from scratch.")

    # 7. Split dataset into train/validation.
    data_train, data_valid, label_train, label_valid = train_test_split(
        data, label, test_size=0.1, shuffle=True
    )
    del data, label

    # 8. Create training and validation generators
    train_generator = triplet_net.create_generator(batch_size, dev_range, data_train, label_train)
    valid_generator = triplet_net.create_generator(batch_size, dev_range, data_valid, label_valid)

    # 9. Define optimizer and LR scheduler.
    optimizer = optim.Adam(triplet_net.parameters(), lr=2e-4)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=10, verbose=True, optimizer=optimizer)

    best_val_loss = float('inf')
    best_model = None

    # 10. Training loop
    for epoch in range(500):
        triplet_net.train()
        train_loss = 0
        # ---- Training steps ----
        for step, (inputs, _) in enumerate(train_generator):
            inputs = [inp.to(device) for inp in inputs]
            optimizer.zero_grad()
            loss = triplet_net(*inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Because we created our generator to iterate over all data,
            # break after covering everything once
            if step >= len(data_train) // batch_size:
                break
        train_loss /= (len(data_train) // batch_size)

        # ---- Validation steps ----
        triplet_net.eval()
        val_loss = 0
        with torch.no_grad():
            for step, (inputs, _) in enumerate(valid_generator):
                inputs = [inp.to(device) for inp in inputs]
                loss = triplet_net(*inputs)
                val_loss += loss.item()
                if step >= len(data_valid) // batch_size:
                    break
        val_loss /= (len(data_valid) // batch_size)

        # Check if current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = triplet_net
            os.makedirs(best_model_path, exist_ok=True)
            torch.save(
                best_model.state_dict(),
                os.path.join(best_model_path, 'best_model.pth')
            )
            print(f"Epoch {epoch + 1}: New best model saved with val_loss {best_val_loss:.4f}")

        # Adjust learning rate if needed
        last_lr = reduce_lr.get_last_lr()
        reduce_lr.step(val_loss)

        print(
            f"Epoch {epoch + 1}/500, "
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"learning rate: {last_lr}"
        )

    return best_model  # or return None if you prefer

def train_feature_extractor_hcf(
    file_path='LoRa_RFFI_dataset/dataset/Train/dataset_training_aug.h5',
    dev_range=np.arange(0, 30, dtype=int),
    pkt_range=np.arange(0, 1000, dtype=int),
    snr_range=np.arange(20, 40),
    best_model_path='vq_vae_models/five_stage_models/hcf/stage_2/',
    pretrained_model_path='vq_vae_models/five_stage_models/hcf/stage_1/best_classifier_hcf_snr_20_40_60.07.pth'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load IQ samples + labels
    raw_data, label = LoadDataset().load_iq_samples(file_path, dev_range, pkt_range)

    # 2. Add AWGN
    data_noisy = awgn(raw_data, snr_range)

    # 3. Hand-crafted RF features
    extra_features = []
    for sig in data_noisy:
        amp_imb, _, CFO, LO_leak = compute_rf_features(sig)
        extra_features.append([amp_imb, CFO, LO_leak])
    extra_features = torch.tensor(np.array(extra_features), dtype=torch.float32)

    # 4. Channel‑independent spectrograms
    spec_data = ChannelIndSpectrogram().channel_ind_spectrogram(data_noisy)
    spec_data = torch.tensor(spec_data, dtype=torch.float32).permute(0, 3, 1, 2)

    # 5. Hyper‑params
    margin     = 1
    batch_size = 128

    # 6. Model
    triplet_net = TripletNet_hcf(spec_data.shape, margin).to(device)

    # 7. Load pretrained encoder weights (if provided)
    if pretrained_model_path and os.path.isfile(pretrained_model_path):
        print(f'Loading pretrained weights: {pretrained_model_path}')
        from models import VQVAE_no_q_hcf  # adjust import if needed
        pre = VQVAE_s_no_q_hcf((1, 102, 62), 128, 30, 30, 30, 0.1)
        pre.load_state_dict(torch.load(pretrained_model_path, map_location=device))

        new_sd = {}
        for k, v in pre.encoder.state_dict().items():
            new_sd[k.replace('encoder.', '')] = v
        for k, v in pre.extra_fc.state_dict().items():
            new_sd[f'extra_fc.{k}'] = v
        for k, v in pre.concat_fc.state_dict().items():
            new_sd[f'concat_fc.{k}'] = v
        triplet_net.embedding_net.load_state_dict(new_sd, strict=False)
        print('Pretrained encoder loaded.')

    # 8. Train / validation split
    data_tr, data_val, extra_tr, extra_val, lbl_tr, lbl_val = train_test_split(
        spec_data, extra_features, label, test_size=0.1, shuffle=True
    )
    del raw_data, data_noisy, spec_data, extra_features, label

    # 9. Generators (assume you have create_generator)
    train_gen = triplet_net.create_generator(batch_size, dev_range, data_tr, extra_tr, lbl_tr)
    valid_gen = triplet_net.create_generator(batch_size, dev_range, data_val, extra_val, lbl_val)
    sil_loader = DataLoader(
        RFSingleDataset(data_val, extra_val, lbl_val),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )


    # 10. Optimiser / scheduler
    optimiser = optim.Adam(triplet_net.parameters(), lr=2e-4)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.2, patience=10, verbose=True)

    best_val_loss = float('inf')
    best_sil_score = float('inf')
    best_model_sd = None

    # 11. Training loop
    for epoch in range(500):
        # ---- train ----
        triplet_net.train()
        train_loss = 0
        for step, (inp, _) in enumerate(train_gen):
            inp = [t.to(device) for t in inp]
            optimiser.zero_grad()
            loss = triplet_net(*inp)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            if step >= len(data_tr) // batch_size:
                break
        train_loss /= (len(data_tr) // batch_size)

        # ---- validate ----
        triplet_net.eval()
        val_loss   = 0
        lat_list   = []
        lab_list   = []
        with torch.no_grad():
            for step, (inp, labs) in enumerate(valid_gen):
                inp  = [t.to(device) for t in inp]
                labs = labs.to(device)
                loss = triplet_net(*inp)
                val_loss += loss.item()
                if step >= len(data_val) // batch_size:
                    break
        val_loss /= (len(data_val) // batch_size)
        print(
            f'Epoch {epoch+1:3d}/500 | '
            f'Train {train_loss:.4f} | Val {val_loss:.4f}'
        )
         # ---------------------- silhouette score ---------------
        latents = []
        labels  = []
        with torch.no_grad():
            for spec_batch, extra_batch, lbl_batch in sil_loader:
                spec_batch  = spec_batch.to(device)
                extra_batch = extra_batch.to(device)
                emb = triplet_net.encode(spec_batch, extra_batch)
                latents.append(emb.cpu())
                labels.append(lbl_batch)
        lat_all = torch.cat(latents, 0).numpy()
        lab_all = torch.cat(labels, 0).numpy().ravel()
        if np.unique(lab_all).size > 1:
            sil = silhouette_score(lat_all, lab_all)
        else:
            sil = float('nan')   # should not happen with stratified split
        print(f'Silhoutte score : {sil:.4f}') 
        # save best on val loss
        if val_loss < best_val_loss or (val_loss == best_val_loss and sil > best_sil_score):
            best_sil_score = sil
            best_val_loss = val_loss
            best_model_sd = triplet_net.state_dict()
            os.makedirs(best_model_path, exist_ok=True)
            torch.save(best_model_sd, os.path.join(best_model_path, 'best_hcf_model.pth'))
            print(f'  ↳  saved new best model (val_loss {best_val_loss:.4f})')

        last_lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

    # return best model
    if best_model_sd is not None:
        triplet_net.load_state_dict(best_model_sd)
    return triplet_net

    




if __name__ == '__main__':

    # Specifies what task the program runs for. 
    # 'Train'/'Classification'/'Rogue Device Detection'
    run_for = 'Train_hcf'

    if run_for == 'Train':

        # Train an RFF extractor.
        feature_extractor = train_feature_extractor()
        # Save the trained model.
        feature_extractor.save('Extractor.keras')
    if run_for == 'Train_hcf':

        # Train an RFF extractor.
        feature_extractor = train_feature_extractor_hcf()
        # Save the trained model.
        feature_extractor.save('Extractor.keras')


    
