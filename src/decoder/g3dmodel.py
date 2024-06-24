import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder.utils import ResBlock3D

class G3D(nn.Module):
    def __init__(self):
        super(G3D, self).__init__()
        self.resblock1 = ResBlock3D(96, 96)
        self.downsample1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.resblock2 = ResBlock3D(96, 192)
        self.downsample2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.resblock3 = ResBlock3D(192, 384)
        self.resblock4 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.resblock5 = ResBlock3D(384, 768)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.resblock6 = ResBlock3D(768+384, 384)  # Concatenate with output from resblock3
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.resblock7 = ResBlock3D(384+192, 192)  # Concatenate with output from resblock2
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.resblock8 = ResBlock3D(192+96, 96)  # Concatenate with output from resblock1
        self.conv_out = ResBlock3D(96, 96)
        self.to("cuda" if torch.cuda.is_available() else "cpu")

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
        x = self.upsample3(x)
        x = torch.cat([x, x1], dim=1)  # Concatenate along the channel dimension
        x = self.resblock8(x)
        x = self.conv_out(x)
        return x