import torch
import torch.nn as nn
import torch.nn.functional as F
from .fuse_block import FuseBlock7, FuseBlock8
from .unet_freq import Unet2D_FNO

class UNet(nn.Module):
    def __init__(
        self, 
        n_channels: int, 
        n_classes: int, 
        bilinear : bool=False,
        trunc_mode_stages: list = None,
        use_sobel_stages: list = None,
        patch_based_stages: list = None,
        patch_size_stages: list= None,
        factorize_mode_stages: list = None,
        use_attn_stages: list = None,
        type_grid: str = None,
        fuse_block: int = None,
        freq_fuse_type: list = None,
        use_fno: bool=False
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.trunc_mode_stages = trunc_mode_stages
        self.use_sobel_stages = use_sobel_stages
        self.patch_based_stages = patch_based_stages
        self.patch_size_stages = patch_size_stages
        self.factorize_mode_stages = factorize_mode_stages
        self.use_attn_stages = use_attn_stages
        self.type_grid = type_grid
        self.fuse_block = fuse_block
        self.freq_fuse_type = freq_fuse_type
        self.use_fno = use_fno

        self._ds_ch = 1024  # Added for downsampled channel tracking

        self.inc = DoubleConv(n_channels, 64)
        
        if self.use_fno:
            # Added fuse blocks
            self.fuse = nn.ModuleList()
            if self.fuse_block == 7:
                get_fuse_block = FuseBlock7
            elif self.fuse_block == 8:
                get_fuse_block = FuseBlock8
            for m in [
                get_fuse_block(128, num_heads=4),
                get_fuse_block(256, num_heads=4),
                get_fuse_block(512, num_heads=4),
                get_fuse_block(1024 // (2 if bilinear else 1), num_heads=4),
            ]:
                self.fuse.append(m)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes) # linear layer
        if self.use_fno:
            # Added frequency domain U-Net
            self.unet_freq = Unet2D_FNO(
                in_channels=n_channels, 
                num_channels=[64, 128, 256, 512, 1024 // factor],
                target_size=[256, 256],  # Assuming 256x256 output size
                trunc_mode_stages=trunc_mode_stages,
                use_sobel_stages=use_sobel_stages,
                patch_based_stages=patch_based_stages,
                patch_size_stages=patch_size_stages,
                factorize_mode_stages=factorize_mode_stages,
                use_attn_stages = use_attn_stages,
                type_grid=type_grid
            )
    @property
    def ds_ch(self):
        return self._ds_ch
    def forward(self, x):
        if self.use_fno:
            # Added frequency domain processing
            fre_x1_s, fre_h_s = self.unet_freq(x)

        x1 = self.inc(x)
        xs = [x1]

        if "enc" in self.freq_fuse_type and self.use_fno:
            for (conv, fuse, fre_h) in zip(
                [self.down1, self.down2, self.down3, self.down4],
                self.fuse,
                fre_h_s
            ):
                xs.append(fuse(conv(xs[-1]), fre_h))
        else:
            for conv in [self.down1, self.down2, self.down3, self.down4]:
                xs.append(conv(xs[-1]))
        x = xs[-1]

        if "dec" in self.freq_fuse_type and self.use_fno:
            x = self.up1(x, xs[3], fre_h_s[-1], self.fuse[-1])
            x = self.up2(x, xs[2], fre_h_s[-2], self.fuse[-2])
            x = self.up3(x, xs[1], fre_h_s[-3], self.fuse[-3])
            x = self.up4(x, xs[0], fre_h_s[-4], self.fuse[-4])
        else:
            x = self.up1(x, xs[3])
            x = self.up2(x, xs[2])
            x = self.up3(x, xs[1])
            x = self.up4(x, xs[0])

        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, x_freq=None, fuse_block=None):
        if x_freq is not None and fuse_block is not None:
            x1 = fuse_block(x1, x_freq)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
