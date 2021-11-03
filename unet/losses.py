# -*- coding: utf-8 -*-
# ---------------------
from pathlib import Path

import piq
import torch
import torchvision
from torch import nn
from torch.nn.modules.loss import _Loss
import pytorch_msssim
from correction_dataset import WoodCorrectionDataset


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1, device='cuda'):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std).to(device)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(device) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean).to(device) / std
        self.requires_grad = False
        self.device = device


class VGGLoss(_Loss):

    def __init__(self, conv_index='2_2', rgb_range=1, device='cuda'):
        super(VGGLoss, self).__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('2_2') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('5_4') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.mean_shift = MeanShift(rgb_range, vgg_mean, vgg_std, device=device)
        self.vgg = self.vgg.to(device)
        self.mean_shift = self.mean_shift.to(device)

    def get_vgg_features(self, x):
        x = self.mean_shift(x)
        x = self.vgg(x)
        return x

    def forward(self, y_pred, y_true):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor

        vgg_sr = self.get_vgg_features(y_pred)
        with torch.no_grad():
            vgg_hr = self.get_vgg_features(y_true.detach())

        loss = nn.functional.mse_loss(vgg_sr, vgg_hr)
        return loss

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class WoodCorrectionLoss(_Loss):

    def __init__(self, vgg_high_w, ms_ssim_w, vgg_low_w, device, verbose=False):
        # type: (float, float, float, str, bool) -> None
        super().__init__()

        self.ms_ssim = piq.MultiScaleSSIMLoss(kernel_size=7, data_range=255).to(device)
        self.ms_ssim_w = ms_ssim_w

        self.vgg_low = piq.ContentLoss(layers=["relu2_2"], replace_pooling=True).to(device) # VGGLoss(conv_index='2_2', device=device).to(device)
        self.vgg_low_w = vgg_low_w

        self.vgg_high = piq.ContentLoss(layers=["relu5_1"], replace_pooling=True).to(device) # VGGLoss(conv_index='5_4', device=device).to(device)
        self.vgg_high_w = vgg_high_w

        self.weights = [self.ms_ssim_w, self.vgg_low_w, self.vgg_high_w]

        self.verbose = verbose

        assert sum(self.weights) != 0, 'at least one weight must be != than 0'

    def forward(self, y_pred, y_true):
        # type: (torch.Tensor, torch.Tensor) -> tuple

        vgg_high_l = 0 if self.vgg_high_w == 0 else self.vgg_high_w * self.vgg_high(y_pred, y_true)
        ms_ssim_l = 0 if self.ms_ssim_w == 0 else self.ms_ssim_w * self.ms_ssim(y_pred, y_true)
        vgg_low_l = 0 if self.vgg_low_w == 0 else self.vgg_low_w * self.vgg_low(y_pred, y_true)

        total = sum([vgg_high_l, ms_ssim_l, vgg_low_l])

        if self.verbose:
            print(f"vgg_high: {float(vgg_high_l)}, vgg_low: {float(vgg_low_l)}, mssim: {float(ms_ssim_l)}")

        return total, vgg_high_l, ms_ssim_l, vgg_low_l


if __name__ == "__main__":
    ds = WoodCorrectionDataset(
        dataset_path=Path("../dataset/Legni02@resize_16x"),
        cut_size_h_w=(128, 256),
        max_shift=15,
        min_shift=0,
        test_mode=False
    )

    loss = WoodCorrectionLoss(vgg_high_w=7, ms_ssim_w=9, vgg_low_w=1, verbose=True, device='cuda')

    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset=ds,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    for couple in dl:
        misaligned = couple[0][0]
        aligned = couple[1][0]

        my_loss = loss.forward(torch.unsqueeze(misaligned, dim=0), torch.unsqueeze(aligned, dim=0))
