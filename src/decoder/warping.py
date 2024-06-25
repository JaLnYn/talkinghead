import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder.utils import ResBlock3D, ResBlock3D_Adaptive
import os


class FlowField(nn.Module):
    def __init__(self):
        super(FlowField, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv1x1 = nn.Conv2d(512, 2048, kernel_size=1).to(device)

        # reshape the tensor from [batch_size, 2048, height, width] to [batch_size, 512, 4, height, width], effectively splitting the channels into a channels dimension of size 512 and a depth dimension of size 4.
        self.reshape_layer = lambda x: x.view(-1, 512, 4, *x.shape[2:]).to(device)

        self.resblock1 = ResBlock3D_Adaptive(in_channels=512, out_channels=256).to(device)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2)).to(device)
        self.resblock2 = ResBlock3D_Adaptive( in_channels=256, out_channels=128).to(device)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2)).to(device)
        self.resblock3 =  ResBlock3D_Adaptive( in_channels=128, out_channels=64).to(device)
        self.upsample3 = nn.Upsample(scale_factor=(1, 2, 2)).to(device)
        self.resblock4 = ResBlock3D_Adaptive( in_channels=64, out_channels=32).to(device)
        self.upsample4 = nn.Upsample(scale_factor=(1, 2, 2)).to(device)
        self.conv3x3x3 = nn.Conv3d(32, 3, kernel_size=3, padding=1).to(device)
        self.gn = nn.GroupNorm(1, 3).to(device)
        self.tanh = nn.Tanh().to(device)
        self.lin = nn.Linear(512, 512).to(device)
    
#    @profile
    def forward(self, zs): # 
        zs = self.lin(zs) # this is the adaptive matrix
        zs.unsqueeze_(-1).unsqueeze_(-1)    
        x = self.conv1x1(zs)
        x = self.reshape_layer(x)
        x = self.upsample1(self.resblock1(x))
        x = self.upsample2(self.resblock2(x))
        x = self.upsample3(self.resblock3(x))
        x = self.upsample4(self.resblock4(x))
        x = self.conv3x3x3(x)
        x = self.gn(x)
        x = F.relu(x)
        x = self.tanh(x)

        # Assertions for shape and values
        assert x.shape[1] == 3, f"Expected 3 channels after conv3x3x3, got {x.shape[1]}"

        return x
    
def compute_rotation_matrix(rotation):
    """
    Computes the rotation matrix from rotation angles.
    
    Args:
        rotation (torch.Tensor): The rotation angles (in degrees) of shape (batch_size, 3).
        
    Returns:
        torch.Tensor: The rotation matrix of shape (batch_size, 3, 3).
    """
    # Assumes rotation is a tensor of shape (batch_size, 3), representing rotation angles in degrees
    rotation_rad = rotation * (torch.pi / 180.0)  # Convert degrees to radians

    cos_alpha = torch.cos(rotation_rad[:, 0])
    sin_alpha = torch.sin(rotation_rad[:, 0])
    cos_beta = torch.cos(rotation_rad[:, 1])
    sin_beta = torch.sin(rotation_rad[:, 1])
    cos_gamma = torch.cos(rotation_rad[:, 2])
    sin_gamma = torch.sin(rotation_rad[:, 2])

    # Compute the rotation matrix using the rotation angles
    zero = torch.zeros_like(cos_alpha)
    one = torch.ones_like(cos_alpha)

    R_alpha = torch.stack([
        torch.stack([one, zero, zero], dim=1),
        torch.stack([zero, cos_alpha, -sin_alpha], dim=1),
        torch.stack([zero, sin_alpha, cos_alpha], dim=1)
    ], dim=1)

    R_beta = torch.stack([
        torch.stack([cos_beta, zero, sin_beta], dim=1),
        torch.stack([zero, one, zero], dim=1),
        torch.stack([-sin_beta, zero, cos_beta], dim=1)
    ], dim=1)

    R_gamma = torch.stack([
        torch.stack([cos_gamma, -sin_gamma, zero], dim=1),
        torch.stack([sin_gamma, cos_gamma, zero], dim=1),
        torch.stack([zero, zero, one], dim=1)
    ], dim=1)

    # Combine the rotation matrices
    rotation_matrix = torch.matmul(R_alpha, torch.matmul(R_beta, R_gamma))

    return rotation_matrix

def compute_rt_warp(rotation, translation, invert=False, grid_size=64):
    """
    Computes the rotation/translation warpings (w_rt).
    
    Args:
        rotation (torch.Tensor): The rotation angles (in degrees) of shape (batch_size, 3).
        translation (torch.Tensor): The translation vector of shape (batch_size, 3).
        invert (bool): If True, invert the transformation matrix.
        
    Returns:
        torch.Tensor: The resulting transformation grid.
    """
    # Compute the rotation matrix from the rotation parameters
    rotation_matrix = compute_rotation_matrix(rotation)

    # Create a 4x4 affine transformation matrix
    affine_matrix = torch.eye(4, device=rotation.device).repeat(rotation.shape[0], 1, 1)

    # Set the top-left 3x3 submatrix to the rotation matrix
    affine_matrix[:, :3, :3] = rotation_matrix

    # Set the first three elements of the last column to the translation parameters
    affine_matrix[:, :3, 3] = translation

    # Invert the transformation matrix if needed
    if invert:
        affine_matrix = torch.inverse(affine_matrix)

    # # Create a grid of normalized coordinates 
    grid = F.affine_grid(affine_matrix[:, :3], (rotation.shape[0], 1, grid_size, grid_size, grid_size), align_corners=False)
    # # Transpose the dimensions of the grid to match the expected shape
    grid = grid.permute(0, 4, 1, 2, 3)
    return grid

class WarpingGenerator(nn.Module):
    def __init__(self, num_channels=512):
        super(WarpingGenerator, self).__init__()
        self.flowfield = FlowField()
        self.num_channels = num_channels ### TODO 3

#    @profile
    def forward(self, zd_sum, Rd, td):

        w_em_c2d = self.flowfield(zd_sum)

        # Compute rotation/translation warping
        # w_rt_c2d = compute_rt_warp(Rd, td, invert=False, grid_size=64)
        zeros = torch.zeros((zd_sum.shape[0], 3))
        w_rt_c2d = compute_rt_warp(zeros, zeros, invert=False, grid_size=64)

         # Resize w_em_c2d to match w_rt_c2d
        w_em_c2d_resized = F.interpolate(w_em_c2d, size=w_rt_c2d.shape[2:], mode='trilinear', align_corners=False)

        # w_c2d = w_rt_c2d + w_em_c2d_resized
        w_c2d = w_em_c2d_resized

        return w_c2d
    
    def save_model(self, path='./models/portrait/decoder/'):
        """
        Save the model parameters to the specified path.
        
        Args:
        model (torch.nn.Module): The PyTorch model to save.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Saving model to {path}")
        torch.save(self.state_dict(), path)
    
    def load_model(self, path='./models/portrait/decoder/'):
        """
        Load the model parameters from the specified path into the model.
        
        Args:
        model (torch.nn.Module): The PyTorch model into which the parameters are loaded.
        """
        print(f"Loading model from {path}")
        self.load_state_dict(torch.load(path))
