import sys

import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn import functional as F

from .ExtractParts import Encoder, channel_reduction
from .CBAM import CBAM
from .SEblock import SEblock
from .generate_mask_process import *

class ExtractModel(nn.Module) :
    def __init__(self, args, device):
        super(ExtractModel, self).__init__()

        self.device = device
        self.image_size = args.image_size
        self.angle = args.angle
        self.length = args.length
        self.num_enc = args.k_clusters
        self.preserve_range = args.preserve_range

        self.idxx, self.idxy = get_small_region(self.image_size, self.angle, self.length, self.preserve_range)

        self.encs = nn.ModuleList([Encoder() for _ in range(self.num_enc)])
        # self.encs = Encoder()
        # print(len(self.encs))
        self.channel_reduction = channel_reduction()
        self.attention = CBAM(self.num_enc * 128, 16)
        # self.attention = SEblock(self.num_enc * 128, 16)

    def forward(self, x):
        patterns = self.extract(x)

        out = []

        for i in range(self.num_enc):
            patternFeatureMap = self.encs[i](patterns[i].unsqueeze(1))
            # patternFeatureMap = self.encs(patterns[i].unsqueeze(1))
            out_ = self.channel_reduction(patternFeatureMap)
            out.append(out_)

        out = torch.cat(out, dim=1)
        out = self.attention(out)
        if torch.isnan(out).any():
            print("Find nan")
            sys.exit()
        out = nn.AdaptiveAvgPool2d(1)(out).squeeze()
        # out = nn.AdaptiveMaxPool2d(1)(out).squeeze()
        out = out.reshape(-1, self.num_enc, out.size()[1] // self.num_enc)
        out = torch.mean(out, dim=2)
        out = F.softmax(out, dim=1) * self.num_enc

        out = (out - torch.mean(out, dim=1, keepdim=True)) / torch.std(out, dim=1, keepdim=True) + 1

        return out  # (?, number of encoder)

    def extract(self, x):
        x_fft = fft.fftshift(fft.fft2(x))
        x_spectrum = torch.abs(x_fft)

        self.clustered_idx, self.X, self.labels = fourier_intensity_extraction(x_spectrum, self.idxx, self.idxy, self.num_enc, self.image_size)
        patterns = torch.empty((self.num_enc, x_fft.size(0), x_fft.size(1), x_fft.size(2))).to(self.device, dtype=torch.float32)

        for i, (idxx, idxy) in enumerate(self.clustered_idx):
            mask = torch.zeros(x_fft.size(1), x_fft.size(2)).to(self.device)
            mask[idxx, idxy] = 1
            temp = torch.empty_like(x_fft)
            for j in range(len(x_fft)):
                temp[j] = x_fft[j] * mask
            patterns[i] = temp
        patterns = torch.abs(fft.ifft2(fft.ifftshift(patterns)))

        return patterns