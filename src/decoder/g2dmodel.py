import torch
import torch.nn as nn

from src.encoder.utils import ResBlock2D    

class G2D(nn.Module):
    def __init__(self):
        super(G2D, self).__init__()
        self.conv1x1 = nn.Conv2d(1536, 512, kernel_size=1)
        self.resblock1 = ResBlock2D(512, 512)
        self.resblock2 = ResBlock2D(512, 512)
        self.resblock3 = ResBlock2D(512, 512)
        self.resblock4 = ResBlock2D(512, 512)
        self.resblock5 = ResBlock2D(512, 512)
        self.resblock6 = ResBlock2D(512, 512)
        self.resblock7 = ResBlock2D(512, 512)
        self.resblock8 = ResBlock2D(512, 512)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.resblock9 = ResBlock2D(512, 256)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.resblock10 = ResBlock2D(256, 128)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.resblock11 = ResBlock2D(128, 64)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, 96*16, 32, 32)
        x = self.conv1x1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.upsample1(x)
        x = self.resblock9(x)
        x = self.upsample2(x)
        x = self.resblock10(x)
        x = self.upsample3(x)
        x = self.resblock11(x)
        x = self.conv_out(x)
        # x = x[:, :, 6:250, 6:250]
        x = x[:, :, 16:240, 16:240]
        return x