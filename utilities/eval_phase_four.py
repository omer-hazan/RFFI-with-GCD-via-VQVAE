import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import your project modules.
# Adjust the import paths as necessary.
from model_based_final_project.models_model_based import VQVAE, VQVAE_s, VQVAE_no_q, VQVAE_s_no_q, VQVAE_s_no_q_hcf, VQVAE_s_hcf
from hand_crafted_features import compute_rf_features
from model_based_final_project.VQ_VAE_eyal_model_based import load_pretrained_model
from liverpool.closeset.dataset_preparation import LoadDataset, awgn, ChannelIndSpectrogram

def evaluate_model(model, dataloader, criterion, device, use_latents=True):
    """
    Evaluate a given model on the provided dataloader.
    For models with a quantizer, set use_latents=True to call forward with return_latents=True.
    """
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, extra_features, labels in dataloader:
            inputs = inputs.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.flatten()

            # For models with a quantizer, use return_latents=True.
            if use_latents and hasattr(model, 'quantizer'):
                output = model(inputs, extra_features, return_latents=True)
            else:
                output = model(inputs, extra_features)

            # Unpack outputs.
            if len(output) == 4:
                preds, quant_loss, _, _ = output
            else:
                preds, quant_loss, _ = output  # For the non-quantizer model, quant_loss is None

            ce_loss = criterion(preds, labels)
            loss = ce_loss + quant_loss if quant_loss is not None else ce_loss

            val_loss += loss.item()
            correct_val += model.get_accuracy(labels, preds)
            total_val += labels.size(0)

    avg_loss = val_loss / len(dataloader)
    accuracy = 100 * correct_val / total_val
    return avg_loss, accuracy

def main():
    # Force the use of CUDA. This script assumes a CUDA-enabled GPU is available.
    device = torch.device("cuda")
    print("Using device:", device)

    # -------------------------------------------------
    # Configuration for loading pretrained weights.
    # -------------------------------------------------
    config = {
        'model_mode': 'close_set_enc',  # Closed set encoder mode.
        'use_hcf': True,                # HCF is enabled.
        'pretrained_encoder': True,
        'encoder_weights_path_hcf_close_set_enc': 'model_based_final_project/new_stage_5/best_classifier_hcf_85.87.pth',
        'pretrained_classifier': True,
        'classifier_weights_path_hcf_close_set_enc': 'model_based_final_project/new_stage_5/best_classifier_hcf_85.87.pth',
        'use_quantizer': True,
        'pretrained_codebook': True,
        'codebook_weights_path_hcf_close_set_enc': 'model_based_final_project/new_stage_4/kmeans_codebooks/codebook_85_87_size_32.pth',
        'freeze_encoder': False,
        'freeze_classifier': False,
        'freeze_codebook': False
    }

    # -------------------------------------------------
    # Define model hyperparameters.
    # -------------------------------------------------
    input_shape = (1, 102, 62)  # e.g., (channels, height, width) for the spectrogram.
    num_embeddings = 128
    hcf_n = 30
    codebook_size = 32
    num_classes = 30  # Adjust based on your dataset.

    # -------------------------------------------------
    # Instantiate both models.
    # -------------------------------------------------
    # Model with quantizer (VQVAE_s_hcf)
    model_with_quantizer = VQVAE_s_hcf(input_shape, num_embeddings, hcf_n, codebook_size, num_classes).to(device)
    # Model without quantizer (VQVAE_s_no_q_hcf)
    model_no_quantizer = VQVAE_s_no_q_hcf(input_shape, num_embeddings, hcf_n, codebook_size, num_classes).to(device)

    # -------------------------------------------------
    # Load pretrained weights.
    # -------------------------------------------------
    print("Loading weights for model with quantizer...")
    model_with_quantizer = load_pretrained_model(model_with_quantizer, config, device)
    
    print("Loading weights for model without quantizer (skipping codebook)...")
    model_no_quantizer = load_pretrained_model(model_no_quantizer, config, device)
    
    # -------------------------------------------------
    # Data loading and processing.
    # -------------------------------------------------
    dataset_path = 'LoRa_RFFI_dataset/dataset/Train/dataset_training_aug.h5'
    dev_range = range(0, 30)
    pkt_range = range(0, 1000)

    load_dataset_obj = LoadDataset()
    data_train, label_train = load_dataset_obj.load_iq_samples(
        file_path=dataset_path,
        dev_range=dev_range,
        pkt_range=pkt_range
    )

    # Shuffle the dataset.
    indices = np.arange(len(label_train))
    np.random.shuffle(indices)
    data_train = data_train[indices, :]
    label_train = label_train[indices]
    
    # Adjust labels (e.g., making them zero-based).
    label_train = label_train - dev_range[0]
    label_train = torch.tensor(label_train, dtype=torch.long)
    
    # Add noise to improve robustness.
    data_train = awgn(data_train, range(20, 80))
    
    # Convert IQ data to spectrograms.
    channel_spec_obj = ChannelIndSpectrogram()
    data = channel_spec_obj.channel_ind_spectrogram(data_train)
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
    
    # Compute extra RF features (amplitude imbalance, CFO, LO leakage).
    extra_features_list = []
    for signal in data_train:
        amp_imbalance, _, CFO, LO_leakage = compute_rf_features(signal)
        extra_features_list.append([amp_imbalance, CFO, LO_leakage])
    extra_features = np.array(extra_features_list)
    extra_features = torch.tensor(extra_features, dtype=torch.float32)
    
    # Create the dataset and dataloader.
    dataset = TensorDataset(data, extra_features, label_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # -------------------------------------------------
    # Evaluate both models.
    # -------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    print("\nEvaluating model with quantizer...")
    loss_quant, acc_quant = evaluate_model(model_with_quantizer, dataloader, criterion, device, use_latents=True)
    print("Model with quantizer -- Loss: {:.4f}, Accuracy: {:.2f}%".format(loss_quant, acc_quant))
    
    print("\nEvaluating model without quantizer...")
    loss_no_quant, acc_no_quant = evaluate_model(model_no_quantizer, dataloader, criterion, device, use_latents=False)
    print("Model without quantizer -- Loss: {:.4f}, Accuracy: {:.2f}%".format(loss_no_quant, acc_no_quant))
    
    # -------------------------------------------------
    # Output the difference in accuracy.
    # -------------------------------------------------
    acc_difference = acc_no_quant - acc_quant
    print("\nDifference in Accuracy (Non-Quantizer - Quantizer): {:.2f}%".format(acc_difference))

if __name__ == '__main__':
    main()
