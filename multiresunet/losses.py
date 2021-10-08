# -*- coding: utf-8 -*-
# ---------------------
from pathlib import Path

import piq
import torch
import torchvision
from torch import nn
from torch.nn.modules.loss import _Loss
from correction_dataset import WoodCorrectionDataset


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean) / std
        self.requires_grad = False


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
        self.mean_shift = MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg = self.vgg.to(device)
        self.mean_shift.to(device)

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

    def __init__(self, mse_w, ssim_w, vgg_w, device='cuda'):  # mse is L1
        # type: (float, float, float, str) -> None
        super().__init__()

        self.loss_functions = [
            nn.L1Loss(), piq.MultiScaleSSIMLoss(kernel_size=7), VGGLoss(device=device)
        ]

        self.weights = [mse_w, ssim_w, vgg_w]  # vgg is always 0
        assert sum(self.weights) != 0, 'at least one weight must be != than 0'

    def forward(self, y_pred, y_true):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        loss = None
        for i in range(len(self.loss_functions)):
            if self.weights[i] != 0:
                loss_component = self.weights[i] * self.loss_functions[i](y_pred, y_true)
                loss = (loss + loss_component) if (loss is not None) else loss_component

        return loss


if __name__ == "__main__":
    ds = WoodCorrectionDataset(
        dataset_path=Path("../dataset/Legni02@resize_16x"),
        cut_size_h_w=(128, 256),
        max_shift=15,
        min_shift=0,
        test_mode=False
    )

    loss = WoodCorrectionLoss(mse_w=8, ssim_w=13, vgg_w=0)

    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset=ds,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    for img in dl:
        misaligned = img[0]
        aligned = img[1]

        my_loss = loss.forward(misaligned, aligned)
        print(f"Loss evaluated: {my_loss:.2f}")
