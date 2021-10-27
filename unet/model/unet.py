""" Full assembly of the parts to form the complete network """
import torch

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, replace_maxpool_with_stride, out_channels=3, bilinear=True, device='cuda'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.device = device

        self.inc = DoubleConv(n_channels, 64, replace_maxpool_with_stride=False, device=self.device)
        self.down1 = Down(64, 128, replace_maxpool_with_stride=replace_maxpool_with_stride, device=self.device)
        self.down2 = Down(128, 256, replace_maxpool_with_stride=replace_maxpool_with_stride, device=self.device)
        self.down3 = Down(256, 512, replace_maxpool_with_stride=replace_maxpool_with_stride, device=self.device)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, replace_maxpool_with_stride=replace_maxpool_with_stride, device=self.device)
        self.up1 = Up(1024, 512 // factor, bilinear, device=self.device)
        self.up2 = Up(512, 256 // factor, bilinear, device=self.device)
        self.up3 = Up(256, 128 // factor, bilinear, device=self.device)
        self.up4 = Up(128, 64, bilinear, device=self.device)
        self.outc = OutConv(64, out_channels, device=self.device)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    unet = UNet(n_channels=3, out_channels=3, replace_maxpool_with_stride=True, device='cuda:0')
    out = unet(torch.ones(5, 3, 128, 256, device='cuda:0'))
    print(out.size())
