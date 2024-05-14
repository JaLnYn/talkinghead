import torch
import torch.nn as nn
import torch.nn.functional as F
from .g2dmodel import G2D
from .g3dmodel import G3D
from .warping import WarpingGenerator


class FaceDecoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.warping_module_d = WarpingGenerator()
        self.warping_module_s = WarpingGenerator()
        self.g3d = G3D()
        self.g2d = G2D()

    def forward(self, Xs, Xd):
        (v_s, e_s, r_s, z_s) = Xs
        (v_d, e_d, r_d, z_d) = Xd

        warping_mod_s_inp = torch.cat([e_s, r_s, z_s], dim=1)
        print(warping_mod_s_inp.shape)
        w_s = self.warping_module_s(warping_mod_s_inp)

        warping_mod_d_inp = torch.cat([e_s, r_d, z_d], dim=1)

        w_d = self.warping_module_s(warping_mod_d_inp)

        print(w_s.shape)
        print(w_d.shape)
        return None
