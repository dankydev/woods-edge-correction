""" Full assembly of the parts to form the complete network """
import torch.nn as nn

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, out_channels=3, ):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 64)

        self.down4 = Down(64, 128)
        self.down5 = Down(128, 128)
        self.down6 = Down(128, 128)

        self.down7 = Down(128, 256)
        self.down8 = Down(256, 256)
        self.down9 = Down(256, 256)

        self.down10 = Down(256, 512)
        self.down11 = Down(512, 512)
        self.down12 = Down(512, 512)

        self.down12 = Down(512, 1024)
        self.down13 = Down(1024, 1024)
        self.down14 = Down(1024, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 512)
        self.up3 = Up(512, 512)

        self.up4 = Up(512, 256)
        self.up5 = Up(256, 256)
        self.up6 = Up(256, 256)

        self.up7 = Up(256, 128)
        self.up8 = Up(128, 128)
        self.up9 = Up(128, 128)

        self.up10 = Up(128, 64)
        self.up11 = Up(64, 64)
        self.up12 = Up(64, 64)

        self.up13 = Up(64, 32)
        self.up14 = Up(32, 32)

        self.out_conv = nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x10 = self.down9(x9)
        x11 = self.down10(x10)
        x12 = self.down11(x11)
        x13 = self.down12(x12)
        x14 = self.down13(x13)
        x15 = self.down14(x14)

        x = self.up1(x15, x14)
        x = self.up2(x, x13)
        x = self.up3(x, x12)
        x = self.up4(x, x11)
        x = self.up5(x, x10)
        x = self.up6(x, x9)
        x = self.up7(x, x8)
        x = self.up8(x, x7)
        x = self.up9(x, x6)
        x = self.up10(x, x5)
        x = self.up11(x, x4)
        x = self.up12(x, x3)
        x = self.up13(x, x2)
        x = self.up14(x, x1)
        logits = self.out_conv(x)
        return logits


if __name__ == "__main__":
    unet = UNet(n_channels=3, out_channels=3)
    out = unet(torch.ones(3, 3, 128, 256))
    ok = 0
