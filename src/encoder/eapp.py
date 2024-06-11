import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.encoder.utils import ResBlock2D, ResBlock3D



class Eapp(nn.Module):
    def __init__(self):
        super(Eapp, self).__init__()
        self.resize = transforms.Resize((256, 256))
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        
        self.resblock2d_1 = ResBlock2D(64, 128)
        self.pool2d_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.resblock2d_2 = ResBlock2D(128, 512)
        self.pool2d_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.gn = nn.GroupNorm(num_groups=32, num_channels=512)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(512, 1536, kernel_size=1)
        
        self.resblock3d_1 = ResBlock3D(96, 96)
        self.resblock3d_2 = ResBlock3D(96, 96)
        self.resblock3d_3 = ResBlock3D(96, 96)
        self.resblock3d_4 = ResBlock3D(96, 96)
        self.resblock3d_5 = ResBlock3D(96, 96)
        self.resblock3d_6 = ResBlock3D(96, 96)
    
    def forward(self, x):
        x = self.resize(x)
        x = self.initial_conv(x)
        
        x = self.resblock2d_1(x)
        x = self.pool2d_1(x)
        
        x = self.resblock2d_2(x)
        x = self.pool2d_2(x)
        
        x = self.gn(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        
        # Reshape for 3D convolution
        x = x.view(x.size(0), 96, 16, x.size(2), x.size(3))
        
        x = self.resblock3d_1(x)
        x = self.resblock3d_2(x)
        x = self.resblock3d_3(x)
        x = self.resblock3d_4(x)
        x = self.resblock3d_5(x)
        x = self.resblock3d_6(x)
        
        return x

    def save_model(self, path='./models/portrait/eapp.pth'):
        """
        Save the model parameters to the specified path.
        
        Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): Path to the file where the model parameters are saved.
        """
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path='./models/portrait/eapp.pth'):
        """
        Load the model parameters from the specified path into the model.
        
        Args:
        model (torch.nn.Module): The PyTorch model into which the parameters are loaded.
        path (str): Path to the file from which the model parameters are loaded.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f'Model loaded from {path}')

def get_eapp_model(path=None, device='cuda'):
    model = Eapp()
    if path is not None and os.path.exists(path):
        model.load_model(path)
    else:
        print('No model found at path. Creating new model.')
    return model


# Example usage
if __name__ == "__main__":
    model = get_eapp_model()  # Create a new model
    # print(model)

    
    input_data = torch.randn(2, 3, 224, 224).to('cuda').to('cuda')

    print(model(input_data).shape)
    