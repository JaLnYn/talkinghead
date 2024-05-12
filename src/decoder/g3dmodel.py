import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class G3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G3D, self).__init__()
        self.resblock1 = ResBlock3D(in_channels, 96)
        self.downsample1 = nn.Conv3d(96, 196, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.resblock2 = ResBlock3D(196, 196)
        self.downsample2 = nn.Conv3d(196, 384, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.resblock3 = ResBlock3D(384, 384)
        self.resblock4 = ResBlock3D(384, 512)
        self.resblock5 = ResBlock3D(512, 512)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.resblock6 = ResBlock3D(512 + 384, 384)  # Concatenate with output from resblock3
        self.upsample2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.resblock7 = ResBlock3D(384 + 196, 196)  # Concatenate with output from resblock2
        self.resblock8 = ResBlock3D(196 + 96, 96)  # Concatenate with output from resblock1
        self.conv_out = nn.Conv3d(96, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.resblock1(x)
        x = self.downsample1(x1)
        x2 = self.resblock2(x)
        x = self.downsample2(x2)
        x3 = self.resblock3(x)
        x = self.resblock4(x3)
        x = self.resblock5(x)
        x = self.upsample1(x)
        x = torch.cat([x, x3], dim=1)  # Concatenate along the channel dimension
        x = self.resblock6(x)
        x = self.upsample2(x)
        x = torch.cat([x, x2], dim=1)  # Concatenate along the channel dimension
        x = self.resblock7(x)
        x = torch.cat([x, x1], dim=1)  # Concatenate along the channel dimension
        x = self.resblock8(x)
        x = self.conv_out(x)
        return x