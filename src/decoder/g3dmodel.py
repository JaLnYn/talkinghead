import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder.utils import ResBlock3D

class G3D(nn.Module):
    def __init__(self):
        super(G3D, self).__init__()
        self.resblock1 = ResBlock3D(96, 96)
        self.downsample1 = nn.Conv3d(96, 196, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.resblock2 = ResBlock3D(196, 196)
        self.downsample2 = nn.Conv3d(196, 384, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.resblock3 = ResBlock3D(384, 384)
        self.resblock4 = nn.Conv3d(384, 512, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.resblock5 = ResBlock3D(512, 512)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.resblock6 = nn.Conv3d(512 + 384, 384, kernel_size=3, padding=1)  # Concatenate with output from resblock3
        self.resblock65 = ResBlock3D(384, 384)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.resblock7 = nn.Conv3d(384 + 196, 196, kernel_size=3, padding=1)  # Concatenate with output from resblock2
        self.resblock75 = ResBlock3D(196, 196)
        self.upsample3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.resblock8 = nn.Conv3d(196 + 96, 96, kernel_size=3, padding=1)  # Concatenate with output from resblock1
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
        x = self.resblock65(x)
        x = self.upsample2(x)
        x = torch.cat([x, x2], dim=1)  # Concatenate along the channel dimension
        x = self.resblock7(x)
        x = self.resblock75(x)
        x = self.upsample3(x)
        x = torch.cat([x, x1], dim=1)  # Concatenate along the channel dimension
        x = self.resblock8(x)
        x = self.conv_out(x)
        return x

    def save_model(self, path='./models/portrait/decoder/g3d.pth'):
        """
        Save the model parameters to the specified path.
        
        Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): Path to the file where the model parameters are saved.
        """
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path='./models/portrait/decoder/g3d.pth'):
        """
        Load the model parameters from the specified path into the model.
        
        Args:
        model (torch.nn.Module): The PyTorch model into which the parameters are loaded.
        path (str): Path to the file from which the model parameters are loaded.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f'Model loaded from {path}')