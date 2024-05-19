import torch
import torch.nn as nn

class ResBlock3DAdaptive(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock3DAdaptive, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adaptive_params = nn.Linear(in_channels, 2 * out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        print(f"\t{x.shape}")
        params = self.adaptive_params(x.mean(dim=(2, 3, 4)))
        gamma, beta = params.chunk(2, dim=-1)
        out = gamma.view(-1, out.shape[1], 1, 1, 1) * out + beta.view(-1, out.shape[1], 1, 1, 1)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class WarpingGenerator(nn.Module):
    def __init__(self):
        super(WarpingGenerator, self).__init__()
        self.conv1x1 = nn.Conv1d(1, 2048, kernel_size=1)
        # self.resblock1 = ResBlock3DAdaptive(512, 512)
        self.resblock1 = nn.GroupNorm(256, 512)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)
        # self.resblock2 = ResBlock3DAdaptive(256, 256)
        self.resblock2 = nn.GroupNorm(128, 512)
        # self.upsample2 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)
        # self.resblock3 = ResBlock3DAdaptive(128, 128)
        self.resblock3 = nn.GroupNorm(64, 512)
        # self.upsample3 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)
        # self.resblock4 = ResBlock3DAdaptive(64, 64)
        # self.resblock4 = nn.GroupNorm(32, 1024)
        # self.upsample4 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)
        # self.resblock5 = ResBlock3DAdaptive(32, 32)
        self.resblock5 = nn.GroupNorm(32, 1024)
        self.conv_out = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.to('cuda')

    def forward(self, x):
        batch_size = x.shape[0]
        print("l1",x.unsqueeze(1).shape, x.unsqueeze(1).device)
        x = self.conv1x1(x.unsqueeze(1))
        print("l2",x.shape, x.device)
        x = x.reshape(batch_size, 512, 4, 512)
        print("l3", x.shape, x.device)
        x = self.resblock1(x)
        print("l4", x.shape, x.device)
        x = self.upsample1(x)
        print("l5", x.shape, x.device)
        x = self.resblock2(x)
        # print("l6", x.shape, x.device)
        # x = self.upsample2(x)
        # x = self.conv2(x)
        print("l6", x.shape, x.device)
        x = self.resblock3(x)
        # print("l8", x.shape, x.device)
        # x = self.upsample3(x)
        # print("l9", x.shape, x.device)
        # x = self.resblock4(x)
        # print("l10", x.shape, x.device)
        # x = self.upsample4(x)
        # print("l12", x.shape, x.device)
        # x = self.resblock5(x)
        print("l7", x.shape)
        x = self.conv_out(x)
        x = self.tanh(x)
        print("l8", x.shape)
        return x

