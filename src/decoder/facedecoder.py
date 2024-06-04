import torch
import torch.nn as nn
import torch.nn.functional as F
from .g2dmodel import G2D
from .g3dmodel import G3D
from .warping import WarpingGenerator
from src.encoder.utils import compute_rotation


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
        warp_inputs = torch.cat([e_s + z_s, r_s, t_s], dim=1)
        w_s = self.warping_module_s(warp_inputs)
        
        apply_ws = torch.einsum("bdxyz,bwxyz->bdxyz", v_s, w_s)
        g_1 = self.g3d(apply_ws)

        warp_inputs = torch.cat([e_d + z_d, r_d, t_d], dim=1)
        w_d = self.warping_module_d(warp_inputs)

        apply_wd = torch.einsum("bdxyz,bwxyz->bdxyz", g_1, w_d)
        g2 = self.g2d(apply_wd)
        return g2
