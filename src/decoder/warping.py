import torch
import torch.nn as nn

class ResBlock3DAdaptive(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock3DAdaptive, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adaptive_params = nn.Linear(in_channels, 2 * out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        params = self.adaptive_params(x.mean(dim=(2, 3, 4)))
        gamma, beta = params.chunk(2, dim=-1)
        out = gamma.view(-1, out.shape[1], 1, 1, 1) * out + beta.view(-1, out.shape[1], 1, 1, 1)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class WarpingGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WarpingGenerator, self).__init__()
        self.conv1x1 = nn.Conv3d(in_channels, 2048, kernel_size=1)
        self.resblock1 = ResBlock3DAdaptive(2048, 512)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.resblock2 = ResBlock3DAdaptive(256, 256)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.resblock3 = ResBlock3DAdaptive(128, 128)
        self.upsample3 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.resblock4 = ResBlock3DAdaptive(64, 64)
        self.upsample4 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.resblock5 = ResBlock3DAdaptive(32, 32)
        self.conv_out = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1x1(x)
        x = x.reshape(-1, 2048, 1, 1, 1)
        x = self.resblock1(x)
        x = self.upsample1(x)
        x = self.resblock2(x)
        x = self.upsample2(x)
        x = self.resblock3(x)
        x = self.upsample3(x)
        x = self.resblock4(x)
        x = self.upsample4(x)
        x = self.resblock5(x)
        x = self.conv_out(x)
        x = self.tanh(x)
        return x

