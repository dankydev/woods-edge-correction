# -*- coding: utf-8 -*-
# ---------------------
from pathlib import Path

import piq
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from correction_dataset import WoodCorrectionDataset


class WoodCorrectionLoss(_Loss):

    def __init__(self, mse_w, vgg_high_w, ms_ssim_w, vgg_low_w, dists_w, dss_w, device, verbose=False):
        # type: (float, float, float, float, float, float, str, bool) -> None
        super().__init__()

        self.mse = nn.MSELoss()
        self.mse_w = mse_w

        self.ms_ssim = piq.MultiScaleSSIMLoss(kernel_size=7, data_range=255).to(device)
        self.ms_ssim_w = ms_ssim_w

        self.vgg_low = piq.ContentLoss(layers=["relu2_2"], replace_pooling=True).to(device)
        self.vgg_low_w = vgg_low_w

        self.vgg_high = piq.ContentLoss(layers=["relu5_1"], replace_pooling=True).to(device)
        self.vgg_high_w = vgg_high_w

        self.dists = piq.DISTS().to(device)
        self.dists_w = dists_w

        self.dss = piq.DSSLoss().to(device)
        self.dss_w = dss_w

        self.weights = [self.mse_w, self.ms_ssim_w, self.vgg_low_w, self.vgg_high_w, self.dists_w, self.dss_w]

        self.verbose = verbose

        assert sum(self.weights) != 0, 'at least one weight must be != than 0'

    def forward(self, y_pred, y_true):
        # type: (torch.Tensor, torch.Tensor) -> tuple

        mse_l = 0 if self.mse_w == 0 else self.mse_w * self.mse(y_pred, y_true)
        vgg_high_l = 0 if self.vgg_high_w == 0 else self.vgg_high_w * self.vgg_high(y_pred, y_true)
        ms_ssim_l = 0 if self.ms_ssim_w == 0 else self.ms_ssim_w * self.ms_ssim(y_pred, y_true)
        vgg_low_l = 0 if self.vgg_low_w == 0 else self.vgg_low_w * self.vgg_low(y_pred, y_true)
        dists_l = 0 if self.dists_w == 0 else self.dists_w * self.dists(y_pred, y_true)
        dss_l = 0 if self.dss_w == 0 else self.dss_w * self.dss(y_pred, y_true)

        total = sum([mse_l, vgg_high_l, ms_ssim_l, vgg_low_l, dists_l, dss_l])

        if self.verbose:
            print(f"mse: {float(mse_l)}, vgg_high: {float(vgg_high_l)}, vgg_low: {float(vgg_low_l)}, mssim: {float(ms_ssim_l)}, dists: {float(dists_l)}, dss: {float(dss_l)}")

        return total, mse_l, vgg_high_l, ms_ssim_l, vgg_low_l, dists_l, dss_l


if __name__ == "__main__":
    ds = WoodCorrectionDataset(
        dataset_path=Path("../dataset/Legni02@resize_16x"),
        cut_size_h_w=(128, 256),
        max_shift=15,
        min_shift=0,
        test_mode=False
    )

    loss = WoodCorrectionLoss(mse_w=0, vgg_high_w=1, ms_ssim_w=0, vgg_low_w=1, dists_w=0, dss_w=0, verbose=True, device='cuda')

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
