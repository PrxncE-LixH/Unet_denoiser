import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        skip = self.conv(x)
        return self.downsample(skip), skip


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, 1, padding=1),  # 1x1
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UnetStem(nn.Module):
    def __init__(
        self, in_channels, out_channels, features=(48, 96, 192, 384), stem=False
    ):
        super().__init__()

        # Stem: one stride-2 conv at the input. Everything downstream then runs at
        # half resolution
        if stem:
            self.stem = nn.Conv2d(in_channels, features[0], 3, stride=2, padding=1)
            c = features[0]
        else:
            self.stem = None
            c = in_channels

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(c, f))
            self.pools.append(nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1))
            c = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.ups = nn.ModuleList()
        cin = features[-1] * 2
        for f in reversed(features):
            self.ups.append(Up(cin, skip_channels=f, out_channels=f))
            cin = f

        # With a stem, no convolution ever runs at full res.
        if stem:
            self.head = nn.Conv2d(features[0], out_channels * 4, kernel_size=1)
            self.shuffle = nn.PixelShuffle(2)
        else:
            self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.shuffle = nn.Identity()

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)

        skips = []
        for conv, pool in zip(self.downs, self.pools):
            x = conv(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.shuffle(self.head(x))
