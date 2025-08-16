###################################################################################################
###################################################################################################
#################################        Adaptive Codebook        #################################
################################################################################################
###################################################################################################

''' Here the implementation of Learning Multi-Rate Vector Quantization for Remote Deep Inference by
May Malka, Shai Ginzach, and Nir Shlezinger

For further questions: maymal@post.bgu.ac.il
'''

###################################################################################################
###################################################################################################
#################################             Imports             #################################
###################################################################################################
###################################################################################################

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import time
from sklearn.metrics import confusion_matrix, silhouette_score

import os
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import math
import wandb

from models import VQVAE, VQVAE_s, VQVAE_no_q, VQVAE_s_no_q, VQVAE_s_no_q_hcf, VQVAE_s_hcf, VQVAE_no_q_hcf
from hand_crafted_features import compute_rf_features

from liverpool.closeset.dataset_preparation import LoadDataset, awgn, ChannelIndSpectrogram
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR

# SEED = 13

# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################################################################################
###################################################################################################
#################################   Globals & Hyperparameters     #################################
###################################################################################################
###################################################################################################

# logging.basicConfig(level=logging.INFO)
#
# BATCH_SIZE = 16  # Changed from 16
#
# LEARNING_RATE = 1e-4
# EPOCHS = 20
# NUM_EMBED = 16  # Number of vectors in the codebook.
# ARCH = 'CIFAR'
# SIZE = 64  # 64 IS FOR IMAGEWOOF
# # OHAD_WAS_HERE = True
# # Eyal_is_here = False
#
#
# ###################################################################################################
# ###################################################################################################
# #################################         Data Arrangment         #################################
# ###################################################################################################
# ###################################################################################################
#
# if ARCH == 'CIFAR':
#     NUM_CLASSES = 100
#     FEATURES = 81920
#
#
#     def get_test_transforms():
#         test_transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#         return test_transform
#
#
#     def get_train_transforms():
#         transform = transforms.Compose(
#             [transforms.RandomCrop(32, padding=4),
#              transforms.RandomHorizontalFlip(),
#              transforms.ToTensor(),
#              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#         return transform
#
#
#     train_transform = get_train_transforms()
#     test_transform = get_test_transforms()
#
#     ssl._create_default_https_context = ssl._create_unverified_context
#     path = "/tmp/cifar100"
#     trainset = datasets.CIFAR100(root=path, train=True, download=True, transform=train_transform)
#     testset = datasets.CIFAR100(root=path, train=False, download=True, transform=test_transform)
#
#     # Define the size of the subset
#     # train_subset_size = int(len(trainset) * 0.01)
#     # test_subset_size = int(len(testset) * 0.01)
#     #
#     # # Create random indices for the subset
#     # train_indices = np.random.choice(len(trainset), train_subset_size, replace=False)
#     # test_indices = np.random.choice(len(testset), test_subset_size, replace=False)
#     #
#     # # Create the subset
#     # train_subset = Subset(trainset, train_indices)
#     # test_subset = Subset(testset, test_indices)
#
#     # Create
#     #     BATCH_SIZE = 64 DataLoaders for the subsets
#
#     trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
#     testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
#
# elif ARCH == 'IMAGEWOOF':
#     NUM_CLASSES = 10
#     FEATURES = 327680
#     ssl._create_default_https_context = ssl._create_unverified_context
#     dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
#     download_url(dataset_url, '.')
#
#     with tarfile.open('./imagewoof2-160.tgz', 'r:gz') as tar:  #read file in r mode
#         tar.extractall(path='./data')  #extract all folders from zip file and store under folder named data
#
#     data_dir = './data/imagewoof2-160'
#     # print(os.listdir(data_dir))
#     # print(os.listdir('./data/imagewoof2-160/train'))
#     # print(len(os.listdir('./data/imagewoof2-160/train')))
#     classes = ['Golden retriever', 'Rhodesian ridgeback', 'Australian terrier', 'Samoyed', 'Border terrier', 'Dingo',
#                'Shih-Tzu', 'Beagle', 'English foxhound', 'Old English sheepdog']
#
#     train_directory = './data/imagewoof2-160/train'
#     test_directory = './data/imagewoof2-160/val'
#
#     image_size_test = ImageFolder(train_directory, transforms.ToTensor())
#
#     train_tfms = transforms.Compose([transforms.Resize([SIZE, SIZE]), transforms.ToTensor()])
#     test_tfms = transforms.Compose([transforms.Resize([SIZE, SIZE]), transforms.ToTensor()])
#
#     trainset = ImageFolder(data_dir + '/train', train_tfms)
#     testset = ImageFolder(data_dir + '/val', test_tfms)
#
#     classes_dict = dict(zip(os.listdir('./data/imagewoof2-160/train'), classes))
#
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)


###################################################################################################
###################################################################################################
#################################       Adaptive Quantizer        #################################
###################################################################################################
###################################################################################################

# images, labels = next(iter(testset))
#
# # Plot the first 16 images in a 4x4 grid
# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
#
# for i, ax in enumerate(axes.flat):
#     img = images[i].permute(1, 2, 0).numpy()  # Convert to numpy array and rearrange dimensions
#     ax.imshow(img)
#     ax.axis('off')
#
# plt.show()

class AdaptiveVectorQuantizer(nn.Module):
    """
      Implements an adaptive vector quantization scheme. The codebook
      (representation vectors) are learned adaptively from low-resolution
      to high-resolution during training.
    """

    def __init__(self, num_embeddings: int, codebook_size: int, commitment_loss_weight=0.1, proximity_loss_weight=0.33):

        """
        Initialize the AdaptiveVectorQuantizer

        Args:
            num_embeddings (int): Size of the vectors.
            codebook_size (int): Number of codebook vectors.
            commitment_loss_weight (float, optional): Commitment loss parameter. Defaults to 0.1.
            proximity_loss_weight (float, optional): Balancing parameter of the proximity loss. Defaults to 0.33.
        """
        super(AdaptiveVectorQuantizer, self).__init__()

        self.d = num_embeddings  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)  #initialze the codebook

        # Balancing parameter lambda for the commintment loss
        self.commitment_loss_weight = commitment_loss_weight
        self.proximity_loss_weight = proximity_loss_weight

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_data, num_active_vectors, previous_active_vectors):
        """
        Foward pass of the Adaptive Vector Quantizer

        Args:
            inputs (Tensor): Input tensor of size B x C x H x W.
            num_active_vectors (Tensor): Number of active vectors for quantization.
            previous_active_vectors (Tensor): Previous active vectors.

        Returns:
            tuple: Tuple containing quantized vectors, losses, and active codebook vectors.
            Possible Edit: Include avg number of bits per image
        """

        #Input Preparation: Reshaping -> Flattening-> Quantization

        input_data = input_data.permute(0, 2, 3, 1).contiguous()  # Input is rearranged to BxHxWxC for easier processing
        input_shape = input_data.shape

        # Flatten input
        flat_input = input_data.view(-1, self.d)  # input vector is flattened to shape(-1,d)

        # Compute the variance of each segment

        quantized_vectors = []
        losses = []

        """The quantization process is performed for multiple levels of active vectors
        Normalization of the Euclidean distance ruins the performance of the model
        """

        for num_active_levels in range(int(np.log2(num_active_vectors))):
            active_vectors = self.codebook.weight[:pow(2, num_active_levels + 1)]

            # Calculate distance between each input vector and codebook vector

            distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(active_vectors ** 2, dim=1)
                         - 2 * torch.matmul(flat_input, active_vectors.t()))

            """
            Encoding:
            encoding indices are determined by finding the index of the nearest codebook vector
                j = argmin {||z_t - e_k||_2} for  k in {1,2,...,Q}
            """
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(
                1)  # finds the closest reference vector for each input vector.
            encodings = torch.zeros(encoding_indices.shape[0], self.p, device=input_data.device)
            encodings.scatter_(1, encoding_indices, 1)  # encoding indices are converted into one-hot encodings

            # Quantize and unflatten
            """
            The quantized vectors are obtained by a weighted sum of codebook vectors using
             the one-hot encodings.
            """
            quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

            quantized_vectors.append(quantized)

        """ Loss Computation:
        q_latent_loss -> commitment loss: measure the difference between the quantized vectors to the encoded
            input vectors

        e_latent_loss -> alignment loss:

        prox_loss -> proximity loss: encourages nearby codebook vectors to be close to one another

        cb_loss -> overall codebook loss

        """
        for num_active_levels in range(int(np.log2(num_active_vectors))):

            if self.training:
                # Loss
                q_latent_loss = F.mse_loss(quantized_vectors[num_active_levels], input_data.detach())  # commitment loss

                if num_active_levels == 0:
                    prox_loss = 0
                    e_latent_loss = F.mse_loss(quantized_vectors[num_active_levels].detach(),
                                               input_data)  # alignment loss

                elif num_active_levels == 1:
                    e_latent_loss = F.mse_loss(quantized_vectors[num_active_levels].detach(), input_data)
                    prox_loss = (num_active_levels * self.proximity_loss_weight) * F.mse_loss(
                        previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                        active_vectors[:pow(2, num_active_levels + 1) // 2])

                else:
                    e_latent_loss = 0
                    prox_loss = self.proximity_loss_weight * F.mse_loss(
                        previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                        active_vectors[
                        :pow(2, num_active_levels + 1) // 2])  # proximity_loss

                cb_loss = q_latent_loss + self.commitment_loss_weight * e_latent_loss + prox_loss  # codebook loss

                quantized_vectors[num_active_levels] = input_data + (
                        quantized_vectors[num_active_levels] - input_data).detach()  # gradient copying
                quantized_vectors[num_active_levels] = quantized_vectors[num_active_levels].permute(0, 3, 1,
                                                                                                    2).contiguous()

            else:
                # convert quantized from BHWC -> BCHW
                quantized_vectors[num_active_levels] = quantized_vectors[num_active_levels].permute(0, 3, 1,
                                                                                                    2).contiguous()

                # print(f'Distortion: {F.mse_loss(flat_input, quantized_vectors[num_active_levels].view(-1,8))}')
                cb_loss = 0

            losses.append(cb_loss)

        return quantized_vectors, losses, active_vectors

    def compute_encoding_per_section_per_quantization(self, flat_ze_section, quantization_level, ADC_resolution):

        if (quantization_level + 1) > ADC_resolution:
            active_vectors_per_section = self.codebook.weight[:pow(2, ADC_resolution)]
        else:
            active_vectors_per_section = self.codebook.weight[:pow(2, quantization_level + 1)]

        distances_per_section_quantization_level = (torch.sum(flat_ze_section ** 2, dim=1, keepdim=True)
                                                    + torch.sum(active_vectors_per_section ** 2, dim=1)
                                                    - 2 * torch.matmul(flat_ze_section,
                                                                       active_vectors_per_section.t()))

        return torch.argmin(distances_per_section_quantization_level, dim=1).unsqueeze(1)

    def forward_with_split(self, input_data, num_active_vectors, previous_active_vectors, bit_resolutions):

        num_ADCs = len(bit_resolutions)

        #Input Preparation: Reshaping -> Flattening-> Quantization
        input_data = input_data.permute(0, 2, 3, 1).contiguous()  # Input is rearranged to BxHxWxC for easier processing
        input_shape = input_data.shape

        # Flatten input
        flat_input = input_data.view(-1, self.d)  # input vector is flattened to shape(-1,d)
        # flat_input = flat_input + torch.normal(mean=0, std= 1, size = flat_input.shape).to(self.device)  # sanity check added noise

        # Split the tensor into numADCs parts
        split_input = torch.chunk(flat_input, chunks=num_ADCs, dim=0)  # returns list of tensors
        # active_vectors_per_section = [self.codebook.weight[:pow(2, bit_resolutions[kk])] for kk in range(num_ADCs)]

        quantized_vectors = []
        losses = []

        encoding_indices = []
        #Apply l2 norm on each section given bit resolution

        for num_active_levels in range(int(np.log2(num_active_vectors))):
            active_vectors = self.codebook.weight[:pow(2, num_active_levels + 1)]
            encoding_indices_section = []
            for ii in range(num_ADCs):
                # for each ADC bit resolution compute distances per quantization level

                encoding_indices_section.append(self.compute_encoding_per_section_per_quantization
                                                (split_input[ii], num_active_levels, bit_resolutions[ii]))

            encoding_indices = torch.cat(encoding_indices_section, dim=0)
            # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # finds the closest reference vector for each input vector.

            encodings = torch.zeros(encoding_indices.shape[0], self.p, device=input_data.device)
            encodings.scatter_(1, encoding_indices, 1)  # encoding indices are converted into one-hot encodings

            # Quantize and unflatten
            """
            The quantized vectors are obtained by a weighted sum of codebook vectors using
             the one-hot encodings.
            """
            quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
            quantized_vectors.append(quantized)

        # Loss computation

        for num_active_levels in range(int(np.log2(num_active_vectors))):

            if self.training:
                # Loss
                q_latent_loss = F.mse_loss(quantized_vectors[num_active_levels], input_data.detach())  # commitment loss

                if num_active_levels == 0:
                    prox_loss = 0
                    e_latent_loss = F.mse_loss(quantized_vectors[num_active_levels].detach(),
                                               input_data)  # alignment loss


                elif num_active_levels == 1:
                    e_latent_loss = F.mse_loss(quantized_vectors[num_active_levels].detach(), input_data)

                    prox_loss = (num_active_levels * self.proximity_loss_weight) * F.mse_loss(
                        previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                        active_vectors[:pow(2, num_active_levels + 1) // 2])

                else:
                    e_latent_loss = 0
                    prox_loss = self.proximity_loss_weight * F.mse_loss(
                        previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                        active_vectors[
                        :pow(2, num_active_levels + 1) // 2])  # proximity_loss

                cb_loss = q_latent_loss + self.commitment_loss_weight * e_latent_loss + prox_loss  # codebook loss

                quantized_vectors[num_active_levels] = input_data + (
                        quantized_vectors[num_active_levels] - input_data).detach()  # gradient copying
                quantized_vectors[num_active_levels] = quantized_vectors[num_active_levels].permute(0, 3, 1,
                                                                                                    2).contiguous()

            else:
                # convert quantized from BHWC -> BCHW
                quantized_vectors[num_active_levels] = quantized_vectors[num_active_levels].permute(0, 3, 1,
                                                                                                    2).contiguous()
                cb_loss = 0

            losses.append(cb_loss)

        return quantized_vectors, losses, active_vectors

    def forward_with_vqvae(self, input_data, active_vectors):
        """
                Foward pass of the Adaptive Vector Quantizer

                Args:
                    inputs (Tensor): Input tensor of size B x C x H x W.
                    num_active_vectors (Tensor): Number of active vectors for quantization.
                    previous_active_vectors (Tensor): Previous active vectors.

                Returns:
                    tuple: Tuple containing quantized vectors, losses, and active codebook vectors.
                    Possible Edit: Include avg number of bits per image
                """

        # Input Preparation: Reshaping -> Flattening-> Quantization

        input_data = input_data.permute(0, 2, 3, 1).contiguous()  # Input is rearranged to BxHxWxC for easier processing
        input_shape = input_data.shape

        # Flatten input
        flat_input = input_data.view(-1, self.d)  # input vector is flattened to shape(-1,d)

        # Compute the variance of each segment
        num_active_levels = int(math.log2(len(active_vectors)))

        quantized_vectors = []
        losses = []

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(active_vectors ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, active_vectors.t()))

        """
        Encoding:
        encoding indices are determined by finding the index of the nearest codebook vector
            j = argmin {||z_t - e_k||_2} for  k in {1,2,...,Q}
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(
            1)  # finds the closest reference vector for each input vector.
        encodings = torch.zeros(encoding_indices.shape[0], self.p, device=input_data.device)
        encodings.scatter_(1, encoding_indices, 1)  # encoding indices are converted into one-hot encodings

        # Quantize and unflatten
        """
        The quantized vectors are obtained by a weighted sum of codebook vectors using
         the one-hot encodings.
        """
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
        quantized_vectors.append(quantized)

        # Loss computation
        if self.training:
            # Loss
            q_latent_loss = F.mse_loss(quantized_vectors[0], input_data.detach())  # commitment loss

            e_latent_loss = F.mse_loss(quantized_vectors[0].detach(), input_data)  # alignment loss

            cb_loss = q_latent_loss + self.commitment_loss_weight * e_latent_loss  # codebook loss

            quantized_vectors[0] = input_data + (
                    quantized_vectors[0] - input_data).detach()  # gradient copying
            quantized_vectors[0] = quantized_vectors[0].permute(0, 3, 1, 2).contiguous()

        else:
            # convert quantized from BHWC -> BCHW
            quantized_vectors[0] = quantized_vectors[0].permute(0, 3, 1, 2).contiguous()
            cb_loss = 0

        losses.append(cb_loss)

        return quantized_vectors, losses, active_vectors


##################################################################################################
##################################################################################################
#######################################       Model        #######################################
##################################################################################################
##################################################################################################

class AdapCB_Model(nn.Module):
    """
    Implements a model with adaptive vector quantization.
    """

    def __init__(self, num_embeddings, codebook_size, commitment_loss_weight=0.05, lambda_p=0.4, quant=True):
        super(AdapCB_Model, self).__init__()

        self.encoder, self.decoder, self.classifier = self.split_network()
        self.quantizer = AdaptiveVectorQuantizer(num_embeddings, codebook_size, commitment_loss_weight, lambda_p)
        self.quant = quant

    def build_model(self, pretrained=True, fine_tune=True):

        """
                Build the base model.

                Args:
                    pretrained (bool, optional): Use pre-trained weights. Defaults to True.
                    fine_tune (bool, optional): Fine-tune all layers. Defaults to True.
                Returns:
                    nn.Module: Built model.
        """

        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        elif not pretrained:
            print('[INFO]: Not loading pre-trained weights')
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 1], [6, 64, 4, 2], [6, 96, 3, 1],
                                     [6, 160, 3, 1], [6, 320, 1, 1]]
        model = models.mobilenet_v2(pretrained=pretrained, num_classes=1000, width_mult=1,
                                    inverted_residual_setting=inverted_residual_setting)
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in model.parameters():
                params.requires_grad = False

        # change the final classification head, it is trainable,
        model.dropout = nn.Dropout(0.1, inplace=True)
        model.fc = nn.Linear(in_features=FEATURES, out_features=NUM_CLASSES, bias=True)
        return model

    def split_network(self):
        """
        Splitting the Network
            Partitions the MobileNetV2 model into three parts: encoder, decoder, and quantizer

        """
        mobilenetv2 = self.build_model()

        encoder = []
        decoder = []
        classifier = []

        res_stop = 5
        for layer_idx, l in enumerate(mobilenetv2.features):
            if layer_idx <= res_stop:
                encoder.append(l)
            else:
                decoder.append(l)

        classifier.append(mobilenetv2.dropout)
        classifier.append(mobilenetv2.fc)

        Encoder = nn.Sequential(*encoder)
        Decoder = nn.Sequential(*decoder)
        Classifier = nn.Sequential(*classifier)
        return Encoder, Decoder, Classifier

    def get_accuracy(self, gt, preds):
        """
                Calculate the accuracy of predictions.

                Args:
                    ground_truth (Tensor): Ground truth labels.
                    predictions (Tensor): Predicted labels.

                Returns:
                    int: Number of correct predictions.
        """
        pred_vals = torch.max(preds.data, 1)[1]
        batch_correct = (pred_vals == gt).sum()
        return batch_correct

    def normalize(self, inputs):
        """
                Normalize the input.

                Args:
                    inputs (Tensor): Input tensor.

                Returns:
                    Tensor: Normalized input.
        """
        # Calculate the vector's magnitude
        mean = inputs.mean()

        output = inputs - mean
        return output

    def forward(self, inputs, num_active, previous_active_vectors):
        """
                Forward pass of the AdapCB_Model.

                Args:
                    inputs (Tensor): Input tensor.
                    num_active_vectors (Tensor): Number of active vectors for quantization.
                    prev_vectors (Tensor): Previous active vectors.

                Returns:
                    tuple: Tuple containing predictions, quantization loss, active codebook vectors, and original encoder output.
        """
        z_e = self.encoder(inputs)
        z_e = self.normalize(z_e)
        if self.quant == True:
            z_q, vq_loss, actives = self.quantizer.forward(z_e, num_active, previous_active_vectors)
        else:
            z_q, vq_loss, actives = [z_e], [0], None

            # Compute distortion between z_e and z_q
        #
        # print(f' Distortion between z_e and z_q using l2 norm {F.mse_loss(z_e.view(-1, self.quantizer.d).detach() , z_q[0].view(-1, self.quantizer.d).detach())}')

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                                        preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
                                                        preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])

        return preds_list, vq_loss, actives, z_e

    def forward_with_split(self, inputs, num_active_vectors, previous_active_vectors, bit_resolutions):

        z_e = self.encoder(inputs)
        z_e = self.normalize(z_e)
        # encoded signal is partitioned into 4 parts and mixed resolution codebooks are applied
        if self.quant == True:
            z_q, vq_loss, actives = self.quantizer.forward_with_split(z_e, num_active, previous_active_vectors,
                                                                      bit_resolutions)
        else:
            z_q, vq_loss, actives = [z_e], [0], None

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                                        preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
                                                        preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])

        return preds_list, vq_loss, actives, z_e

    def forward_with_vqvae(self, inputs, active_vectors):

        z_e = self.encoder(inputs)
        z_e = self.normalize(z_e)
        # encoded signal is partitioned into 4 parts and mixed resolution codebooks are applied
        if self.quant == True:
            z_q, vq_loss, actives = self.quantizer.forward_with_vqvae(z_e, active_vectors)
        else:
            z_q, vq_loss, actives = [z_e], [0], None

        preds_list = []
        for vecs in range(len(z_q)):
            z_q_actives = z_q[vecs]
            preds_list.append(self.decoder(z_q_actives))
            preds_list[vecs] = preds_list[vecs].reshape(preds_list[vecs].shape[0],
                                                        preds_list[vecs].shape[1] * preds_list[vecs].shape[2] *
                                                        preds_list[vecs].shape[3])
            preds_list[vecs] = self.classifier(preds_list[vecs])

        return preds_list, vq_loss, actives, z_e



###################################################################################################
###################################################################################################
##################################       Training Function        #################################
###################################################################################################
###################################################################################################


def train_fr(model, optimizer, criterion, trainloader, valloader, device, epochs=500, save_path="best_model.pth"):
    wandb.watch(model, log="all")  # Log gradients and parameters
    base_path = "plots"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(base_path, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    start_time = time.time()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_num, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            if labels.ndim > 1:  # If labels are one-hot encoded, convert to class indices
                labels = labels.flatten()

            # Forward pass
            preds, quant_loss = model(inputs)

            # Compute loss
            ce_loss = criterion(preds, labels)
            if quant_loss:
                loss = ce_loss + quant_loss
            else:
                loss = ce_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct_train += model.get_accuracy(labels, preds)
            total_train += labels.size(0)

        train_loss = epoch_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc})
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.2f}%, LR: {scheduler.get_last_lr()}")

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                if labels.ndim > 1:
                    labels = labels.flatten()

                preds, quant_loss = model(inputs)
                ce_loss = criterion(preds, labels)
                if quant_loss:
                    loss = ce_loss + quant_loss
                else:
                    loss = ce_loss

                val_loss += loss.item()
                correct_val += model.get_accuracy(labels, preds)
                total_val += labels.size(0)

                predicted_labels = torch.argmax(preds, dim=1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted_labels.cpu().numpy())

            val_loss /= len(valloader)
            val_acc = 100 * correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch [{epoch + 1}], Val Loss: {val_loss:.6f}, Val Accuracy: {val_acc:.2f}%")

        # --- Save best model & compute metrics if improved ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model at Epoch {epoch + 1} with Val Accuracy: {val_acc:.2f}%")

            # Compute confusion matrix & accuracy per class
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            classes = np.unique(all_labels)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            cm_filename = os.path.join(run_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(cm_filename)
            plt.close()
            print(f"Saved confusion matrix to {cm_filename}")

            # Per-class accuracy
            acc_per_class = {}
            for i, cls in enumerate(classes):
                total = np.sum(cm[i, :])
                acc = 100 * cm[i, i] / total if total > 0 else 0
                acc_per_class[cls] = acc

            # Bar chart: per-class accuracy
            plt.figure(figsize=(10, 6))
            plt.bar(list(acc_per_class.keys()), list(acc_per_class.values()), color='skyblue')
            plt.xlabel("Device/Class")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy per Device")
            plt.ylim([0, 100])
            for cls, acc in acc_per_class.items():
                plt.text(cls, acc + 1, f"{acc:.2f}%", ha='center')
            plt.tight_layout()
            bar_filename = os.path.join(run_dir, f"accuracy_per_device_{timestamp}.png")
            plt.savefig(bar_filename)
            plt.close()
            print(f"Saved accuracy per device bar diagram to {bar_filename}")

            # -------------- NEW CODE FOR CODEBOOK USAGE --------------
            # If the model uses a quantizer, track the usage of each code index.
            # We'll do another pass over val_loader to see how often each codeword is used.
            if hasattr(model, 'quantizer'):
                # Initialize a counter for each codeword
                num_codewords = model.quantizer.num_embeddings  # or your codebook dimension
                codeword_counts = np.zeros(num_codewords, dtype=int)

                # Collect code indices on the validation set
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs = inputs.to(device)
                        # In your model's forward(), ensure you can retrieve the code indices.
                        # For example, you might have something like:
                        #   preds, quant_loss, code_indices = model(inputs, return_indices=True)
                        # or store code_indices as an attribute inside the model.quantizer.
                        
                        # If your forward call doesn't return indices, you'll need to modify
                        # the model/quantizer to give them back. Below is an example pattern:
                        preds, quant_loss, code_indices = model(inputs, return_indices=True)

                        # code_indices should be a tensor of shape [batch_size] or [batch_size, ...]
                        # Flatten if needed:
                        code_indices = code_indices.view(-1).cpu().numpy()
                        for idx in code_indices:
                            codeword_counts[idx] += 1

                # Plot usage as a bar chart
                plt.figure(figsize=(10, 6))
                plt.bar(range(num_codewords), codeword_counts, color='tab:purple')
                plt.xlabel("Codeword Index")
                plt.ylabel("Usage Count")
                plt.title("Codeword Usage Frequency")
                usage_path = os.path.join(run_dir, f"codeword_usage_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(usage_path)
                plt.close()
                print(f"Saved codeword usage bar chart to {usage_path}")
            # -------------- END NEW CODE FOR CODEBOOK USAGE --------------

        # Step the scheduler
        scheduler.step(val_loss)
    
    duration = time.time() - start_time
    print(f"Training completed in {duration / 3600:.2f} hours")
    return train_losses, train_accuracies, val_losses, val_accuracies


def train_fr_hcf(model, optimizer, criterion, trainloader, valloader, device, 
                 epochs=500, save_path="best_model.pth"):
    """
    Train loop for an HCF model. Now tracks silhouette score of latent space
    (z_q) each epoch in the validation phase.
    """
    wandb.watch(model, log="all")  # Log gradients and parameters
    base_path = "plots"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(base_path, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0

    # Example scheduler
    scheduler = CyclicLR(
        optimizer,
        base_lr=5e-7,
        max_lr=5e-4,
        step_size_up=20,
        mode='triangular',
        cycle_momentum=False  # turn off momentum tweaks if using Adam
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=6)
    best_val_loss = 1e6

    for epoch in range(epochs):
        # -------------------- Training --------------------
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_num, (inputs, extra_features, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.flatten()

            # Forward pass
            output = model(inputs, extra_features, return_latents=False)
            # output could be (preds, quant_loss) or (preds, quant_loss, code_indices)
            if len(output) == 2:
                preds, quant_loss = output
            else:
                preds, quant_loss, code_indices = output

            ce_loss = criterion(preds, labels)
            if quant_loss is not None:
                loss = ce_loss + quant_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct_train += model.get_accuracy(labels, preds)
            total_train += labels.size(0)

        train_loss = epoch_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc})
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, "
              f"Train Accuracy: {train_acc:.2f}%, LR: {scheduler.get_last_lr()[0]}")

        # -------------------- Validation --------------------
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # We'll collect latents here to compute silhouette after the loop
        all_labels = []
        all_preds = []
        all_latents = []

        with torch.no_grad():
            for inputs, extra_features, labels in valloader:
                inputs = inputs.to(device)
                extra_features = extra_features.to(device)
                labels = labels.to(device)
                if labels.ndim > 1:
                    labels = labels.flatten()

                # Forward pass with return_latents=True
                output = model(inputs, extra_features, return_latents=True)
                # Now output is (preds, quant_loss, code_indices, z_q)
                if len(output) == 4:
                    preds, quant_loss, code_indices, z_e = output
                else:
                    # fallback if older model version
                    preds, quant_loss = output
                    z_e = None

                ce_loss = criterion(preds, labels)
                if quant_loss is not None:
                    loss = ce_loss + quant_loss
                else:
                    loss = ce_loss

                val_loss += loss.item()
                correct_val += model.get_accuracy(labels, preds)
                total_val += labels.size(0)

                predicted_labels = torch.argmax(preds, dim=1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted_labels.cpu().numpy())

                # Collect latents for silhouette (only if we have z_q)
                if z_e is not None:
                    # Move to CPU numpy for silhouette
                    all_latents.append(z_e.detach().cpu().numpy())

            val_loss /= len(valloader)
            val_acc = 100 * correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch [{epoch + 1}/{epochs}], Val Loss: {val_loss:.6f}, "
                  f"Val Accuracy: {val_acc:.2f}%")

            # ---------- Compute Silhouette Score on z_q ----------
            # Make sure there is more than one class in your batch or silhouette_score will fail
            if len(all_latents) > 0: 
                latents_concat = np.concatenate(all_latents, axis=0)
                labels_concat = np.concatenate(all_labels, axis=0)
                preds_concat = np.concatenate(all_preds, axis=0)
                unique_labels = np.unique(labels_concat)
                if len(unique_labels) > 1:
                    sil_score_true = silhouette_score(latents_concat, labels_concat)
                    wandb.log({"epoch": epoch + 1, "silhouette_score_true": sil_score_true})
                    print(f"Silhouette Score (z_e vs. true labels): {sil_score_true:.4f}")
                    # sil_score_pred = silhouette_score(latents_concat, preds_concat)
                    wandb.log({"epoch": epoch + 1, "silhouette_score_pred": 0})
                    print(f"Silhouette Score (z_e vs. pred labels): {0:.4f}")
                else:
                    print("Silhouette score requires at least 2 distinct labels, skipping.")

        # -------------------- Check if best model --------------------
        if val_acc > best_val_acc or (val_acc == best_val_acc and best_val_loss > val_loss):
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model at Epoch {epoch + 1} with Val Accuracy: {val_acc:.2f}%")

            # Build confusion matrix on the entire val set
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            classes = np.unique(all_labels)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            cm_filename = os.path.join(run_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(cm_filename)
            plt.close()
            print(f"Saved confusion matrix to {cm_filename}")

            # Accuracy per class bar chart
            acc_per_class = {}
            for i, cls in enumerate(classes):
                total_cls = np.sum(cm[i, :])
                acc_cls = 100 * cm[i, i] / total_cls if total_cls > 0 else 0
                acc_per_class[cls] = acc_cls

            plt.figure(figsize=(10, 6))
            plt.bar(list(acc_per_class.keys()), list(acc_per_class.values()))
            plt.xlabel("Device/Class")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy per Device")
            plt.ylim([0, 100])
            for cls, acc_c in acc_per_class.items():
                plt.text(cls, acc_c + 1, f"{acc_c:.2f}%", ha='center')
            plt.tight_layout()
            bar_filename = os.path.join(run_dir, f"accuracy_per_device_{timestamp}.png")
            plt.savefig(bar_filename)
            plt.close()
            print(f"Saved accuracy per device bar chart to {bar_filename}")

            # ---- If model has a quantizer, plot code usage ----
            if hasattr(model, 'quantizer'):
                num_codewords = model.quantizer.p  # codebook_size
                codeword_counts = np.zeros(num_codewords, dtype=int)

                with torch.no_grad():
                    for inputs, extra_features, labels in valloader:
                        inputs = inputs.to(device)
                        extra_features = extra_features.to(device)
                        out = model(inputs, extra_features, return_latents=False)
                        if len(out) == 3:
                            _, _, code_indices = out
                            code_indices = code_indices.view(-1).cpu().numpy()
                            for idx in code_indices:
                                codeword_counts[idx] += 1

                # Plot code usage
                plt.figure(figsize=(10, 6))
                plt.bar(range(num_codewords), codeword_counts)
                plt.xlabel("Codeword Index")
                plt.ylabel("Usage Count")
                plt.title("Codeword Usage Frequency")
                code_usage_file = os.path.join(run_dir, f"codeword_usage_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(code_usage_file)
                plt.close()
                print(f"Saved codeword usage bar chart to {code_usage_file}")

        # Step the scheduler
        # scheduler.step(val_loss)
        scheduler.step()

    duration = time.time() - start_time
    print(f"Training completed in {duration / 3600:.2f} hours")
    return train_losses, train_accuracies, val_losses, val_accuracies



#################################################################################################
#################################################################################################
##############################      Train Adaptive Codebook       ###############################
#################################################################################################
#################################################################################################

def setup_config():
    """
    Define training parameters and mode switches.
    Adjust these flags and paths to select your desired training permutation.
    """
    config = {
        # Pretrained component flags:
        'pretrained_encoder': True,     # Load pretrained encoder weights?
        'pretrained_classifier': False,  # Load pretrained classifier weights?
        'pretrained_codebook': True,    # Load pretrained codebook weights? (Only used if use_quantizer is True)
        'load_codebook_from_encoder': False,
        
        # Freeze options (if using pretrained parts):
        'freeze_encoder': True,          # Freeze encoder parameters after loading pretrained weights?
        'freeze_classifier': False,      # Freeze classifier parameters after loading pretrained weights?
        'freeze_codebook': False,        # Freeze codebook parameters after loading pretrained weights?
        'freeze_hcf_params': True,
        
        # Architecture options:
        'use_hcf': True,                 # Include hand-crafted features (HCF)?
        'use_quantizer': True,          # Use a model variant with a quantizer/codebook?
        'model_size': 'small',
        
        'model_mode': 'close_set_enc',         # Options: 'close_set_enc', 'triplet', 'ss'
        'add_noise': False,
        'snr_range': range(20, 80),       # SNR range for AWGN
        'dev_range': range(30, 40),
        'pkt_range': range(0, 100),
        'path_to_lora_train': 'LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5',
        'input_shape': (1, 102, 62),      # Example spectrogram input shape: (channels, height, width)
        'num_hcf_features': 30,
        'num_classes': 10,
        'codebook_size': 128,
        'batch_size': 32,
        'epochs': 1000,
        'num_embeddings': 128,
        'seed':1111,
        
        # Pretrained weight file paths for model_mode 'close_set_enc':
        'encoder_weights_path_close_set_enc': "vq_vae_models/encoder_classifier/close_set_enc_128_83.37.pth",
        'encoder_weights_path_hcf_close_set_enc': "vq_vae_models/five_stage_models/hcf/stage_5/best_classifier_hcf_84.53.pth",
        'classifier_weights_path_close_set_enc': "vq_vae_models/encoder_classifier/close_set_enc_128_83.37.pth",
        'classifier_weights_path_hcf_close_set_enc': "vq_vae_models/five_stage_models/hcf/stage_5/best_classifier_hcf_84.53.pth",
        'codebook_weights_path_close_set_enc': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_close_set_128.pth",
        'codebook_weights_path_hcf_close_set_enc': "vq_vae_models/five_stage_models/hcf/stage_4_codebook/codebook_dev_30_40_84_53_size_128.pth",

        # Pretrained weight file paths for model_mode 'triplet':
        'encoder_weights_path_triplet': "vq_vae_models/encoder_classifier/encoder_weights_triplet.pth",
        'encoder_weights_path_hcf_triplet': "vq_vae_models/triplet_net/best_hcf_model.pth",
        'classifier_weights_path_triplet': "vq_vae_models/encoder_classifier/classifier_weights_triplet.pth",
        'classifier_weights_path_hcf_triplet': "vq_vae_models/encoder_classifier/hcf/best_classifier_79_enc_froz_triplet.pth",
        'codebook_weights_path_triplet': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_triplet.pth",
        'codebook_weights_path_hcf_triplet': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_hcf_triplet.pth",

        # Pretrained weight file paths for model_mode 'ss' (self-supervised):
        'encoder_weights_path_ss': "vq_vae_models/self_supervised_net_awgn/encoder_weights_ss.pth",
        'encoder_weights_path_hcf_ss': "vq_vae_models/self_supervised_net_awgn/hcf/encoder_weights_ss.pth",
        'classifier_weights_path_ss': "vq_vae_models/self_supervised_net_awgn/classifier_weights_ss.pth",
        'classifier_weights_path_hcf_ss': "vq_vae_models/self_supervised_net_awgn/hcf/classifier_weights_ss.pth",
        'codebook_weights_path_ss': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_ss.pth",
        'codebook_weights_path_hcf_ss': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_hcf_ss.pth",
    }
    # Set the save path for the trained model.
    if config['use_quantizer']:
        if config['model_mode'] == 'close_set_enc':
            if config['use_hcf']:
                config['save_path'] = "vq_vae_models/five_stage_models/hcf/stage_5/best_classifier_hcf.pth"
            else:
                config['save_path'] = "vq_vae_models/vq_vae_classifier/close_set_enc/best_classifier.pth"
        elif config['model_mode'] == 'triplet':
            if config['use_hcf']:
                config['save_path'] = "vq_vae_models/vq_vae_classifier/triplet_enc/best_classifier_hcf.pth"
            else:
                config['save_path'] = "vq_vae_models/vq_vae_classifier/triplet_enc/best_classifier.pth"
    else:
        if config['model_mode'] == 'close_set_enc':
            if config['use_hcf']:
                if config['pretrained_encoder']:
                    config['save_path'] = "vq_vae_models/five_stage_models/hcf/stage_3/best_classifier_hcf.pth"
                else:
                    config['save_path'] = "vq_vae_models/five_stage_models/hcf/stage_1/best_classifier_hcf.pth"
            else:
                config['save_path'] = "vq_vae_models/encoder_classifier/close_set_enc/best_classifier.pth"
        elif config['model_mode'] == 'triplet':
            if config['use_hcf']:
                config['save_path'] = "vq_vae_models/encoder_classifier/triplet_enc/best_classifier_hcf.pth"
            else:
                config['save_path'] = "vq_vae_models/encoder_classifier/triplet_enc/best_classifier.pth"
    return config

def setup_data(config):
    # 1. Load IQ samples and labels.
    LoadDatasetObj = LoadDataset()
    data_train, label_train = LoadDatasetObj.load_iq_samples(
        file_path=config['path_to_lora_train'],
        dev_range=config['dev_range'],
        pkt_range=config['pkt_range']
    )

    # 2. Shuffle and adjust labels.
    index = np.arange(len(label_train))
    np.random.shuffle(index)
    data_train = data_train[index, :]
    label_train = label_train[index]
    label_train = label_train - config['dev_range'][0]
    label_train = torch.tensor(label_train, dtype=torch.long)

    # 3. Add AWGN.
    if config['add_noise']:
        data_train = awgn(data_train, config['snr_range'])

    # 4. Convert IQ samples to channel-independent spectrograms.
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    spectrogram_data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_train)
    spectrogram_data = torch.tensor(spectrogram_data, dtype=torch.float32).permute(0, 3, 1, 2)

    # 5. Create dataset.
    if config['use_hcf']:
        extra_features_list = []
        for signal in data_train:
            # compute_rf_features returns (amp imbalance, phase imbalance, CFO, LO leakage)
            amp_imbalance, _, CFO, LO_leakage = compute_rf_features(signal)
            extra_features_list.append([amp_imbalance, CFO, LO_leakage])
        extra_features = np.array(extra_features_list)
        extra_features = torch.tensor(extra_features, dtype=torch.float32)
        dataset = TensorDataset(spectrogram_data, extra_features, label_train)
    else:
        dataset = TensorDataset(spectrogram_data, label_train)

    # 6. Split dataset into training and validation sets.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader

def build_model(config, device):
    """
    Build the appropriate model based on the configuration flags.
    """
    if config['model_size'] == 'small':
        if config['use_quantizer']:
            # Models with quantizer.
            if config['use_hcf']:
                model = VQVAE_s_hcf(
                    config['input_shape'],
                    num_embeddings=config['num_embeddings'],
                    hcf_n = config['num_hcf_features'],
                    codebook_size=config['codebook_size'],
                    num_classes=config['num_classes']
                )
            else:
                model = VQVAE_s(
                    config['input_shape'],
                    num_embeddings=config['num_embeddings'],
                    codebook_size=config['codebook_size'],
                    num_classes=config['num_classes']
                )
        else:
            # Models without quantizer.
            if config['use_hcf']:
                model = VQVAE_s_no_q_hcf(
                    config['input_shape'],
                    num_embeddings=config['num_embeddings'],
                    hcf_n = 30,
                    codebook_size=config['codebook_size'],
                    num_classes=config['num_classes']
                )
            else:
                model = VQVAE_s_no_q(
                    config['input_shape'],
                    num_embeddings=config['num_embeddings'],
                    codebook_size=config['codebook_size'],
                    num_classes=config['num_classes']
                )
    elif config['model_size'] == 'big':
        if config['use_quantizer']:
            # Models with quantizer.
            if config['use_hcf']:
                model = VQVAE_hcf(
                    config['input_shape'],
                    num_embeddings=config['num_embeddings'],
                    hcf_n = config['num_hcf_features'],
                    codebook_size=config['codebook_size'],
                    num_classes=config['num_classes']
                )
            else:
                model = VQVAE(
                    config['input_shape'],
                    num_embeddings=config['num_embeddings'],
                    codebook_size=config['codebook_size'],
                    num_classes=config['num_classes']
                )
        else:
            # Models without quantizer.
            if config['use_hcf']:
                model = VQVAE_no_q_hcf(
                    config['input_shape'],
                    num_embeddings=config['num_embeddings'],
                    hcf_n = 30,
                    codebook_size=config['codebook_size'],
                    num_classes=config['num_classes']
                )
            else:
                model = VQVAE_no_q(
                    config['input_shape'],
                    num_embeddings=config['num_embeddings'],
                    codebook_size=config['codebook_size'],
                    num_classes=config['num_classes']
                )
    else:
        print('model size did npt configured correctly')
    model = model.to(device)
    return model


def load_pretrained_model(model, config, device):
    """
    Load pretrained weights according to the configuration.
    Each component (encoder, classifier, and codebook) is loaded only if its
    corresponding flag is True. Then, if a freeze flag is set, the loaded part's 
    parameters are frozen. The file paths are chosen based on both the model_mode 
    and whether HCF is used.

    This version also handles 'TripletNet_hcf' style checkpoints that store
    CNN + HCF in 'embedding_net.*' and re-maps them to model.encoder, model.extra_fc, 
    and model.concat_fc as needed.
    """

    mode = config['model_mode']
    use_hcf = config['use_hcf']

    # -------------------------------------------------
    # Determine which file paths to load
    # -------------------------------------------------
    encoder_path = classifier_path = codebook_path = None

    if mode == 'close_set_enc':
        if use_hcf:
            encoder_path = config.get('encoder_weights_path_hcf_close_set_enc')
            classifier_path = config.get('classifier_weights_path_hcf_close_set_enc')
            codebook_path = config.get('codebook_weights_path_hcf_close_set_enc')
        else:
            encoder_path = config.get('encoder_weights_path_close_set_enc')
            classifier_path = config.get('classifier_weights_path_close_set_enc')
            codebook_path = config.get('codebook_weights_path_close_set_enc')

    elif mode == 'triplet':
        if use_hcf:
            encoder_path = config.get('encoder_weights_path_hcf_triplet')
            classifier_path = config.get('classifier_weights_path_hcf_triplet')
            codebook_path = config.get('codebook_weights_path_hcf_triplet')
        else:
            encoder_path = config.get('encoder_weights_path_triplet')
            classifier_path = config.get('classifier_weights_path_triplet')
            codebook_path = config.get('codebook_weights_path_triplet')

    elif mode == 'ss':
        if use_hcf:
            encoder_path = config.get('encoder_weights_path_hcf_ss')
            classifier_path = config.get('classifier_weights_path_hcf_ss')
            codebook_path = config.get('codebook_weights_path_hcf_ss')
        else:
            encoder_path = config.get('encoder_weights_path_ss')
            classifier_path = config.get('classifier_weights_path_ss')
            codebook_path = config.get('codebook_weights_path_ss')

    # -------------------------------------------------
    # 1) Load the encoder weights + HCF submodules
    # -------------------------------------------------
    if config['pretrained_encoder'] and encoder_path is not None and os.path.exists(encoder_path):
        state_dict = torch.load(encoder_path, map_location=device)

        # We'll split the weights that belong to:
        #  - encoder (CNN)
        #  - extra_fc (HCF)
        #  - concat_fc (HCF)
        encoder_dict = {}
        extra_fc_dict = {}
        concat_fc_dict = {}
        old_classifier_dict = {}
        # Some old models store everything under "embedding_net.*"
        # Others might store them under "encoder.*"
        # We'll allow either prefix.
        possible_prefixes = ["embedding_net.", "encoder."]

        for key, val in state_dict.items():
            # Strip off any recognized prefix
            sub_key = key
            for pfx in possible_prefixes:
                if sub_key.startswith(pfx):
                    sub_key = sub_key[len(pfx):]
                    break
        
            # Handle quantizer parameters conditionally
            if sub_key.startswith("quantizer."):
                if config.get('load_codebook_from_encoder', False):
                    quantizer_subkey = sub_key[len("quantizer."):]
                    if 'quantizer_dict' not in locals():
                        quantizer_dict = {}
                    quantizer_dict[quantizer_subkey] = val
                # Ensure quantizer keys are never loaded into encoder_dict
                continue  # <- Crucial fix!
        
            # Continue assigning to correct submodules
            if sub_key.startswith("extra_fc."):
                new_key = sub_key[len("extra_fc."):]  
                extra_fc_dict[new_key] = val
            elif sub_key.startswith("concat_fc."):
                new_key = sub_key[len("concat_fc."):]
                concat_fc_dict[new_key] = val
            elif sub_key.startswith("classifier."):
                new_key = sub_key[len("classifier."):]
                old_classifier_dict[new_key] = val
            else:
                encoder_dict[sub_key] = val
        
        # EXTRA SAFETY STEP: Remove quantizer keys explicitly again
        encoder_dict = {k: v for k, v in encoder_dict.items() if not k.startswith("quantizer.")}
        
        # Now safely load encoder weights
        model.encoder.load_state_dict(encoder_dict, strict=True)
        print("Pretrained encoder loaded from", encoder_path)
        
        # Load HCF if applicable
        if use_hcf:
            model.extra_fc.load_state_dict(extra_fc_dict, strict=True)
            model.concat_fc.load_state_dict(concat_fc_dict, strict=True)
            print("Loaded HCF layers (extra_fc, concat_fc) from", encoder_path)
        
        # Load quantizer explicitly after encoder loading:
        if config['use_quantizer'] and hasattr(model, 'quantizer'):
            if config.get('load_codebook_from_encoder', False):
                if 'quantizer_dict' in locals():
                    model.quantizer.load_state_dict(quantizer_dict, strict=True)
                    print("Quantizer loaded from encoder checkpoint:", encoder_path)
                else:
                    print("Quantizer parameters not found in encoder checkpoint.")
            elif config['pretrained_codebook']:
                if codebook_path is not None and os.path.exists(codebook_path):
                    lbg_codebook = torch.load(codebook_path, map_location=device)
                    model.quantizer.codebook.weight.data.copy_(lbg_codebook)
                    print("Quantizer codebook loaded from separate file:", codebook_path)
                else:
                    print("Codebook pretrained flag set but file not found:", codebook_path)

    # -------------------------------------------------
    # 2) Load the classifier weights
    # -------------------------------------------------
    if config['pretrained_classifier'] and classifier_path is not None and os.path.exists(classifier_path):
        classifier_state = torch.load(classifier_path, map_location=device)
        # Typically, your new classifier is model.classifier, so let's parse that out
        # If the old checkpoint keys are "classifier.*", strip that prefix.
        classifier_dict = {
            k.replace("classifier.", ""): v
            for k, v in classifier_state.items()
            if k.startswith("classifier.")
        }
        model.classifier.load_state_dict(classifier_dict, strict=True)
        print("Pretrained classifier loaded from", classifier_path)
    else:
        if config['pretrained_classifier']:
            print("Classifier pretrained flag set but file not found:", classifier_path)

    # -------------------------------------------------
    # 3) Load codebook (if using a quantizer)
    # -------------------------------------------------
    if config['use_quantizer'] and config['pretrained_codebook'] and hasattr(model, 'quantizer') and not config.get('load_codebook_from_encoder', False):
        if codebook_path is not None and os.path.exists(codebook_path):
            lbg_codebook = torch.load(codebook_path, map_location=device)
            # The codebook is just an nn.Embedding's weight
            model.quantizer.codebook.weight.data.copy_(lbg_codebook)
            print("Pretrained codebook loaded from", codebook_path)
        else:
            print("Codebook pretrained flag set but file not found:", codebook_path)

    # -------------------------------------------------
    # 4) Freeze parameters if requested
    # -------------------------------------------------
    if config.get('freeze_encoder', False):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")
    if config.get('use_hcf', False) and config.get('freeze_hcf_params', False):
        for param in model.extra_fc.parameters():
            param.requires_grad = False
        for param in model.concat_fc.parameters():
            param.requires_grad = False
        print("HCF parameters frozen.")

    if config.get('freeze_classifier', False):
        for param in model.classifier.parameters():
            param.requires_grad = False
        print("Classifier parameters frozen.")

    if config['use_quantizer'] and config.get('freeze_codebook', False) and hasattr(model, 'quantizer'):
        for param in model.quantizer.parameters():
            param.requires_grad = False
        print("Codebook parameters frozen.")

    return model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For (slightly) more reproducible behavior, but might degrade performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    # Initialize configuration.
    config = setup_config()
    set_seed(config['seed'])
    # (Optional) Print config to verify settings.
    print("Training configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Set up Weights & Biases.
    wandb.login(key="f873a533ed8359c89bb63389601e9439b0f2f853")
    wandb.init(project="vq-vae-classifier", entity='https-www-bgu-ac-il-')
    
    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data loaders.
    train_loader, val_loader = setup_data(config)
    
    # Build the model.
    model = build_model(config, device)
    
    # Load pretrained parts as requested (and freeze parts if specified).
    model = load_pretrained_model(model, config, device)
    
    # Set optimizer and loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    
    # Choose the appropriate training loop based on whether HCF is used.
    if config['use_hcf']:
        train_losses, train_accuracies, val_losses, val_accuracies = train_fr_hcf(
            model, optimizer, criterion, train_loader, val_loader, device,
            epochs=config['epochs'], save_path=config['save_path']
        )
    else:
        train_losses, train_accuracies, val_losses, val_accuracies = train_fr(
            model, optimizer, criterion, train_loader, val_loader, device,
            epochs=config['epochs'], save_path=config['save_path']
        )
    
    print("Training completed.")

if __name__ == "__main__":
    main()