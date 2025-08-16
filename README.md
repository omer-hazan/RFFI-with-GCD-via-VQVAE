# RFFI with Generalized Class Discovery via Learning-Aided Vector Quantization

This repository contains the implementation of our approach for **Radio Frequency Fingerprinting Identification (RFFI)** with **open-set recognition** and **generalized class discovery (GCD)**.  
Our method extends **Vector Quantized Variational Autoencoder (VQ-VAE)** models with learning-aided vector quantization, clustering, and codebook expansion mechanisms to enable the discovery of novel devices in wireless networks.  

---

## 📖 Overview

Traditional RFFI systems are designed for **closed-set classification**, where all device classes are known during training. In realistic scenarios, however, new (rogue) devices appear after deployment.  
This work addresses that challenge by:

- Training a **VQ-VAE classifier** where the decoder is replaced with an MLP.  
- Utilizing **Mahalanobis distance** in both triplet-loss training and vector quantization for better discrimination.  
- Leveraging **histogram features** of inverse codeword distances (e.g., entropy, peak relations, mean, median) for **open-set recognition**.  
- Employing **DBSCAN clustering** and **centroid matching** to discover and assign new classes.  
- Supporting **extended codebooks**, where rogue samples can form new labeled codewords for downstream classification.  

This framework is among the first to apply **generalized class discovery** to RFFI using a learning-aided quantization approach.  

---

## 📊 Dataset & Benchmark

We use the **LoRa dataset** introduced in [Shen et al., 2022](https://arxiv.org/abs/2201.XXXX) (*Towards scalable and channel-robust radio frequency fingerprint identification for LoRa*).  

- **Devices 1–30**: Training the feature extractor (with AWGN augmentation as in the benchmark).  
- **Devices 31–40**: Enrollment phase (legitimate devices).  
- **Devices 41–45**: Rogue devices for **open-set recognition** and **class discovery**.  

⚠️ Note: The original benchmark does not include novel class discovery; our extensions fill this gap.  

---

## ⚙️ Features

- **Closed-set classification** using VQ-VAE.  
- **Open-set recognition** with histogram-based metrics.  
- **Generalized class discovery** via clustering (DBSCAN, spectral methods, BIC-based estimates).  
- **Codebook expansion** with new rogue-derived codewords.  
- **Visualization tools** (PCA, clustering plots, purity metrics).  

---
