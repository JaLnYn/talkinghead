import torch
import torch.nn as nn
import torch.nn.functional as F
from .g2dmodel import G2D
from .g3dmodel import G3D
from .warping import WarpingGenerator
from src.encoder.utils import compute_rotation


class FaceDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.warping_module_d = WarpingGenerator()
        self.warping_module_s = WarpingGenerator()
        self.g3d = G3D()
        self.g2d = G2D()

    def forward(self, Xs, Xd):
        (v_s, e_s, r_s, t_s, z_s) = Xs # v_s*z_s-e_s 
        (v_d, e_d, r_d, t_d, z_d) = Xd
        print("vshape", v_s.shape)
        print("eshape", e_s.shape)
        print("zshape", z_s.shape)
        # print("vshape",v_s[1].shape)
        # print("vshape",v_s[2].shape)
        # print("vshape",v_s[3].shape)

        # new_in = torch.cat([e_s, r_s, z_s], dim=1)
        # print(new_in.shape)
        # warping_mod_s_inp = torch.cat([e_s, r_s, z_s], dim=1)
        # print(warping_mod_s_inp.shape)
        warp_inputs = torch.cat([e_s + z_s, r_s, t_s], dim=1)
        w_s = self.warping_module_s(warp_inputs)
        # print(w_s.shape)
        # print(v_s.shape)
        # print(t_s.shape)
        print(w_s.shape, v_s.shape)
        apply_ws = torch.einsum("bdxyz,bwxyz->bdxyz", v_s, w_s)
        print(apply_ws.shape)
        g_1 = self.g3d(apply_ws)
        print("DoneG1", g_1.shape)

        warp_inputs = torch.cat([e_d + z_d, r_d, t_d], dim=1)
        w_d = self.warping_module_d(warp_inputs)

        apply_wd = torch.einsum("bdxyz,bwxyz->bdxyz", g_1, w_d)
        print("apply_wd", apply_wd.shape)
        g2 = self.g2d(apply_wd)

        print(g2.shape)
        return None
