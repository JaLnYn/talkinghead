import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .g2dmodel import G2D
from .g3dmodel import G3D
from .warping import WarpingGenerator
from src.encoder.utils import compute_rotation

def apply_warping_field(v, warp_field):
    B, C, D, H, W = v.size()

    device = v.device

    # Resize warp_field to match the dimensions of v
    warp_field = F.interpolate(warp_field, size=(D, H, W), mode='trilinear', align_corners=True)

    # Create a meshgrid for the canonical coordinates
    d = torch.linspace(-1, 1, D, device=device)
    h = torch.linspace(-1, 1, H, device=device)
    w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack((grid_w, grid_h, grid_d), dim=-1)  # Shape: [D, H, W, 3]

    # Add batch dimension and repeat the grid for each item in the batch
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # Shape: [B, D, H, W, 3]

    # Apply the warping field to the grid
    warped_grid = grid + warp_field.permute(0, 2, 3, 4, 1)  # Shape: [B, D, H, W, 3]

    # Normalize the grid to the range [-1, 1]
    normalization_factors = torch.tensor([W-1, H-1, D-1], device=device)
    warped_grid = 2.0 * warped_grid / normalization_factors - 1.0

    # Apply grid sampling
    v_canonical = F.grid_sample(v, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

    return v_canonical

class FaceDecoder(nn.Module):
    def __init__(self):
        super(FaceDecoder, self).__init__()
        self.warping_module_d = WarpingGenerator()
        self.warping_module_s = WarpingGenerator()
        self.g3d = G3D()
        self.g2d = G2D()

    def forward(self, Xs, Xd):
        (v_s, e_s, r_s, t_s, z_s) = Xs # v_s*z_s-e_s 
        (v_d, e_d, r_d, t_d, z_d) = Xd
        w_s = self.warping_module_s(e_s + z_s, r_s, t_s)
        
        # apply_ws = torch.einsum("bdxyz,bwxyz->bdxyz", v_s, w_s)
        apply_ws = apply_warping_field(v_s, w_s)
        g_1 = self.g3d(apply_ws)

        w_d = self.warping_module_d(e_s + z_d, r_d, t_d)

        # apply_wd = torch.einsum("bdxyz,bwxyz->bdxyz", g_1, w_d)
        apply_wd = apply_warping_field(g_1, w_d)
        g2 = self.g2d(apply_wd)
        return g2

