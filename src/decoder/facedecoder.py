import torch
import torch.nn as nn
import torch.nn.functional as F
from .g2dmodel import G2D
from .g3dmodel import G3D
from .warping import WarpingGenerator


class FaceDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.warping_module_d = WarpingGenerator()
        self.warping_module_s = WarpingGenerator()
        self.g3d = G3D(2, 4)
        self.g2d = G2D(2, 4)

    def forward(self, Xs, Xd):
        (v_s, e_s, r_s, z_s) = Xs
        (v_d, e_d, r_d, z_d) = Xd
        print("eshape",e_s.shape)
        print("rshape",r_s.shape)
        print("zshape",z_s.shape)
        print("vshape",v_s.shape)
        # print("vshape",v_s[1].shape)
        # print("vshape",v_s[2].shape)
        # print("vshape",v_s[3].shape)

        # new_in = torch.cat([e_s, r_s, z_s], dim=1)
        # print(new_in.shape)
        # warping_mod_s_inp = torch.cat([e_s, r_s, z_s], dim=1)
        # print(warping_mod_s_inp.shape)
        w_s = self.warping_module_s(e_s + z_s)
        print(w_s.shape)
        # print(v_s.shape)
        g_1 = torch.einsum('nhwc,nc->nhw', w_s, v_s)
        print(g_1.shape)

        warping_mod_d_inp = torch.cat([e_s, r_d, z_d], dim=1)

        w_d = self.warping_module_s(e_s + z_d)

        print(w_s.shape)
        print(w_d.shape)
        return None
