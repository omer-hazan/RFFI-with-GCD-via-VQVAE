import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Residual block definition
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, first_layer=False):
        super(ResBlock, self).__init__()

        self.first_layer = first_layer
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.first_layer or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Identity loss
def identity_loss(y_true, y_pred):
    return torch.mean(y_pred)

# TripletNet definition
class TripletNet(nn.Module):
    def __init__(self, datashape, alpha):
        super(TripletNet, self).__init__()
        self.datashape = datashape
        self.alpha = alpha
        self.embedding_net = self.feature_extractor()

    def triplet_loss(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        basic_loss = pos_dist - neg_dist + self.alpha
        # loss = torch.clamp(basic_loss, min=0.0)
        loss = basic_loss
        return loss.mean()

    def feature_extractor(self):
        class FeatureExtractor(nn.Module):
            def __init__(self, datashape):
                super(FeatureExtractor, self).__init__()
                self.conv1 = nn.Conv2d(datashape[1], 32, 7, stride=2, padding=3)
                self.resblock1 = ResBlock(32, 32, 3)
                self.resblock2 = ResBlock(32, 32, 3)
                self.resblock3 = ResBlock(32, 64, 3, first_layer=True)
                self.resblock4 = ResBlock(64, 64, 3)
                self.pool = nn.AvgPool2d(2)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(24000, 128)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.resblock1(x)
                x = self.resblock2(x)
                x = self.resblock3(x)
                x = self.resblock4(x)
                x = self.pool(x)
                x = self.flatten(x)
                x = self.fc(x)
                x = F.normalize(x, p=2, dim=1)
                return x

        return FeatureExtractor(self.datashape)

    def forward(self, input_1, input_2, input_3):
        anchor = self.embedding_net(input_1)
        positive = self.embedding_net(input_2)
        negative = self.embedding_net(input_3)
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

    def create_generator(self, batchsize, dev_range, data, label):
        self.data = data
        self.label = label
        self.dev_range = dev_range

        def get_triplet():
            n = a = np.random.choice(self.dev_range)
            while n == a:
                n = np.random.choice(self.dev_range)
            a, p = call_sample(a), call_sample(a)
            n = call_sample(n)
            return a, p, n

        def call_sample(label_name):
            num_sample = len(self.label)
            idx = np.random.randint(num_sample)
            while self.label[idx] != label_name:
                idx = np.random.randint(num_sample)
            return self.data[idx]

        while True:
            list_a, list_p, list_n = [], [], []
            for _ in range(batchsize):
                a, p, n = get_triplet()
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)

            A = torch.tensor(np.array(list_a, dtype='float32'))
            P = torch.tensor(np.array(list_p, dtype='float32'))
            N = torch.tensor(np.array(list_n, dtype='float32'))
            yield [A, P, N], torch.ones(batchsize)

class TripletNet_hcf(nn.Module):
    def __init__(self, datashape, alpha):
        super().__init__()
        self.alpha = alpha
        self.embedding_net = self._build_feature_extractor(datashape)

    # ---------------- triplet loss (unchanged) ----------------
    def triplet_loss(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        return torch.clamp(pos_dist - neg_dist + self.alpha, min=0).mean()

    # ---------------- feature extractor ----------------
    def _build_feature_extractor(self, datashape):
        class FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                C = datashape[1]
                self.conv1 = nn.Conv2d(C, 32, kernel_size=7, stride=2, padding=3)
                self.resblock1 = ResBlock(32, 32, 3)
                self.resblock2 = ResBlock(32, 32, 3)
                self.resblock3 = ResBlock(32, 64, 3, first_layer=True)
                self.resblock4 = ResBlock(64, 64, 3)
                self.pool      = nn.AvgPool2d(2)
                self.fc   = nn.Linear(24000, 128)

                # HCF branch
                self.extra_fc  = nn.Sequential(nn.Linear(3, 30), nn.ReLU())
                self.concat_fc = nn.Sequential(nn.Linear(128 + 30, 128), nn.ReLU())

            def forward(self, x, extra, *, return_latents=False):
                x = F.relu(self.conv1(x))
                x = self.resblock1(x)
                x = self.resblock2(x)
                x = self.resblock3(x)
                x = self.resblock4(x)
                x = self.pool(x)
                x = self.fc(torch.flatten(x, 1))
                x = F.normalize(x, p=2, dim=1)

                extra = self.extra_fc(extra)
                combined = self.concat_fc(torch.cat([x, extra], dim=1))
                combined = F.normalize(combined, p=2, dim=1)  # final embedding

                if return_latents:
                    return combined
                else:
                    return combined  # kept for compatibility

        return FeatureExtractor()

    # ---------------- public helpers ----------------
    def encode(self, spectrogram, extra):
        """Return *only* the L2‑normalised embedding."""
        return self.embedding_net(spectrogram, extra, return_latents=True)

    # standard forward keeps the triplet‑loss behaviour
    def forward(self, anchor, positive, negative, extra_a, extra_p, extra_n):
        emb_a = self.embedding_net(anchor,   extra_a, return_latents=True)
        emb_p = self.embedding_net(positive, extra_p, return_latents=True)
        emb_n = self.embedding_net(negative, extra_n, return_latents=True)
        return self.triplet_loss(emb_a, emb_p, emb_n)

    def forward(self, input_1, extra_1, input_2, extra_2, input_3, extra_3):
        """
        Args:
            input_1, input_2, input_3 (Tensor): CNN inputs (e.g. spectrogram images) for anchor, positive, and negative.
            extra_1, extra_2, extra_3 (Tensor): Hand-crafted feature vectors (shape: [batch, 3]) for anchor, positive, and negative.
        Returns:
            loss (Tensor): Triplet loss computed over the batch.
        """
        anchor = self.embedding_net(input_1, extra_1)
        positive = self.embedding_net(input_2, extra_2)
        negative = self.embedding_net(input_3, extra_3)
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

    def create_generator(self, batchsize, dev_range, data, extra_features, label):
        """
        A generator yielding triplets for training.
        Args:
            batchsize (int): Number of triplets per batch.
            dev_range (iterable): Device range (or label range) to sample from.
            data (np.array): CNN input data (e.g. spectrograms).
            extra_features (np.array): Extra hand-crafted features corresponding to data.
            label (np.array): Labels corresponding to data.
        Yields:
            A tuple: ([A, extra_A, P, extra_P, N, extra_N], target) where each element is a Tensor.
        """
        self.data = data
        self.extra_features = extra_features
        self.label = label
        self.dev_range = dev_range

        def get_triplet():
            a = np.random.choice(self.dev_range)
            n = a
            while n == a:
                n = np.random.choice(self.dev_range)
            a_sample = call_sample(a)
            p_sample = call_sample(a)
            n_sample = call_sample(n)
            return a_sample, p_sample, n_sample

        def call_sample(label_name):
            num_sample = len(self.label)
            idx = np.random.randint(num_sample)
            while self.label[idx] != label_name:
                idx = np.random.randint(num_sample)
            # Return a tuple: (CNN input, extra features)
            return self.data[idx], self.extra_features[idx]

        while True:
            list_a_img, list_p_img, list_n_img = [], [], []
            list_a_extra, list_p_extra, list_n_extra = [], [], []
            for _ in range(batchsize):
                (a_img, a_extra), (p_img, p_extra), (n_img, n_extra) = get_triplet()
                list_a_img.append(a_img)
                list_p_img.append(p_img)
                list_n_img.append(n_img)
                list_a_extra.append(a_extra)
                list_p_extra.append(p_extra)
                list_n_extra.append(n_extra)

            A = torch.tensor(np.array(list_a_img, dtype='float32'))
            extra_A = torch.tensor(np.array(list_a_extra, dtype='float32'))
            P = torch.tensor(np.array(list_p_img, dtype='float32'))
            extra_P = torch.tensor(np.array(list_p_extra, dtype='float32'))
            N = torch.tensor(np.array(list_n_img, dtype='float32'))
            extra_N = torch.tensor(np.array(list_n_extra, dtype='float32'))
            yield [A, extra_A, P, extra_P, N, extra_N], torch.ones(batchsize)

class TripletNet_hcf_big(nn.Module):
    def __init__(self, datashape, alpha):
        super().__init__()
        self.alpha = alpha
        self.embedding_net = self._build_feature_extractor(datashape)

    # ---------------- triplet loss (unchanged) ----------------
    def triplet_loss(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        return torch.clamp(pos_dist - neg_dist + self.alpha, min=0).mean()

    # ---------------- feature extractor ----------------
    def _build_feature_extractor(self, datashape):
        class FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(datashape[1], 32, kernel_size=7, stride=2, padding=3)
                self.resblock1 = ResBlock(32, 32)
                self.resblock2 = ResBlock(32, 64)
                self.resblock3 = ResBlock(64, 64, first_layer=True)
                self.resblock4 = ResBlock(64, 128)
                self.resblock5 = ResBlock(128, 128, first_layer=True)
                self.resblock6 = ResBlock(128, 256)
                self.resblock7 = ResBlock(256, 256, first_layer=True)
                self.resblock8 = ResBlock(256, 256)
                self.avg_pool = nn.AvgPool2d(kernel_size=2)
                self.flatten = nn.Flatten()
                # The 96000 in fc might need adjusting if your input dims change
                self.fc = nn.Linear(96000, 512)

                # HCF branch
                self.extra_fc  = nn.Sequential(nn.Linear(3, 30), nn.ReLU())
                self.concat_fc = nn.Sequential(nn.Linear(512 + 30, 512), nn.ReLU())

            def forward(self, x, extra, *, return_latents=False):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.resblock1(x)
                x = self.resblock2(x)
                x = self.resblock3(x)
                x = self.resblock4(x)
                x = self.resblock5(x)
                x = self.resblock6(x)
                x = self.resblock7(x)
                x = self.resblock8(x)
                x = self.avg_pool(x)
                x = self.flatten(x)
                x = self.fc(x)
                x = F.normalize(x, p=2, dim=1)  # L2 normalization

                extra = self.extra_fc(extra)
                combined = self.concat_fc(torch.cat([x, extra], dim=1))
                combined = F.normalize(combined, p=2, dim=1)  # final embedding

                if return_latents:
                    return combined
                else:
                    return combined  # kept for compatibility

        return FeatureExtractor()
    def encode(self, spectrogram, extra):
        """Return *only* the L2‑normalised embedding."""
        return self.embedding_net(spectrogram, extra, return_latents=True)

    # standard forward keeps the triplet‑loss behaviour
    def forward(self, anchor, positive, negative, extra_a, extra_p, extra_n):
        emb_a = self.embedding_net(anchor,   extra_a, return_latents=True)
        emb_p = self.embedding_net(positive, extra_p, return_latents=True)
        emb_n = self.embedding_net(negative, extra_n, return_latents=True)
        return self.triplet_loss(emb_a, emb_p, emb_n)

    def forward(self, input_1, extra_1, input_2, extra_2, input_3, extra_3):
        """
        Args:
            input_1, input_2, input_3 (Tensor): CNN inputs (e.g. spectrogram images) for anchor, positive, and negative.
            extra_1, extra_2, extra_3 (Tensor): Hand-crafted feature vectors (shape: [batch, 3]) for anchor, positive, and negative.
        Returns:
            loss (Tensor): Triplet loss computed over the batch.
        """
        anchor = self.embedding_net(input_1, extra_1)
        positive = self.embedding_net(input_2, extra_2)
        negative = self.embedding_net(input_3, extra_3)
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

    def create_generator(self, batchsize, dev_range, data, extra_features, label):
        """
        A generator yielding triplets for training.
        Args:
            batchsize (int): Number of triplets per batch.
            dev_range (iterable): Device range (or label range) to sample from.
            data (np.array): CNN input data (e.g. spectrograms).
            extra_features (np.array): Extra hand-crafted features corresponding to data.
            label (np.array): Labels corresponding to data.
        Yields:
            A tuple: ([A, extra_A, P, extra_P, N, extra_N], target) where each element is a Tensor.
        """
        self.data = data
        self.extra_features = extra_features
        self.label = label
        self.dev_range = dev_range

        def get_triplet():
            a = np.random.choice(self.dev_range)
            n = a
            while n == a:
                n = np.random.choice(self.dev_range)
            a_sample = call_sample(a)
            p_sample = call_sample(a)
            n_sample = call_sample(n)
            return a_sample, p_sample, n_sample

        def call_sample(label_name):
            num_sample = len(self.label)
            idx = np.random.randint(num_sample)
            while self.label[idx] != label_name:
                idx = np.random.randint(num_sample)
            # Return a tuple: (CNN input, extra features)
            return self.data[idx], self.extra_features[idx]

        while True:
            list_a_img, list_p_img, list_n_img = [], [], []
            list_a_extra, list_p_extra, list_n_extra = [], [], []
            for _ in range(batchsize):
                (a_img, a_extra), (p_img, p_extra), (n_img, n_extra) = get_triplet()
                list_a_img.append(a_img)
                list_p_img.append(p_img)
                list_n_img.append(n_img)
                list_a_extra.append(a_extra)
                list_p_extra.append(p_extra)
                list_n_extra.append(n_extra)

            A = torch.tensor(np.array(list_a_img, dtype='float32'))
            extra_A = torch.tensor(np.array(list_a_extra, dtype='float32'))
            P = torch.tensor(np.array(list_p_img, dtype='float32'))
            extra_P = torch.tensor(np.array(list_p_extra, dtype='float32'))
            N = torch.tensor(np.array(list_n_img, dtype='float32'))
            extra_N = torch.tensor(np.array(list_n_extra, dtype='float32'))
            yield [A, extra_A, P, extra_P, N, extra_N], torch.ones(batchsize)