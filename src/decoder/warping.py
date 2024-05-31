import torch
import torch.nn as nn
from src.encoder.utils import ResBlock3D

class WarpingGenerator(nn.Module):
    def __init__(self):

        super(WarpingGenerator, self).__init__()
        self.lin1 = nn.Linear(518, 1024)
        self.conv1 = nn.Conv1d(1, 128, kernel_size=1)
        self.resblock1 = ResBlock3D(512, 512)
        self.upsample1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.resblock2 = ResBlock3D(256, 256)
        self.upsample2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.resblock3 = ResBlock3D(128, 128)
        self.conv_out = nn.Conv3d(128, 16, kernel_size=3, padding=1)  # 16 channels for 4x4 transformation matrix
        self.tanh = nn.Tanh()
        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.lin1(x)
        x = self.conv1(x.unsqueeze(1))
        x = x.view(batch_size, 512, 4, 8, -1)
        x = self.resblock1(x)
        x = self.upsample1(x)
        x = self.resblock2(x)
        x = self.upsample2(x)
        x = self.resblock3(x)
        x = self.conv_out(x)
        x = self.tanh(x)
        return x