from src.decoder.facedecoder import FaceDecoder
from src.encoder.emocoder import get_trainable_emonet
from src.encoder.deep3dfacerecon import get_face_recon_model
from src.encoder.hopenet import get_model_hopenet
from src.encoder.arcface import get_model_arcface
import torch
import torch.nn as nn
import dlib

class Portrait(nn.Module):
    def __init__(self):
        super().__init__()

        arcface_model_path = "./models/arcface2/model_ir_se50.pth"
        face3d_model_path = "./models/face3drecon.pth"
        hope_model_path = "./models/hopenet_robust_alpha1.pkl"
        emo_model_path = "./models/emo_path"
        emo_model_path = None

        self.detector = dlib.get_frontal_face_detector()

        self.face3d = get_face_recon_model(face3d_model_path)
        # self.hopenet = get_model_hopenet(hope_model_path)
        self.arcface = get_model_arcface(arcface_model_path)
        self.emodel = get_trainable_emonet(emo_model_path)

        self.decoder = FaceDecoder()

    def forward(self, Xs, Xd):
        # input are images
        coeffs_s = self.face3d(Xs, compute_render=False)

        coef_dict_s = self.face3d.facemodel.split_coeff(coeffs_s)
        v_s = self.face3d.facemodel.compute_shape(coef_dict_s['id'], coef_dict_s['exp'])
        tex_s = self.face3d.facemodel.compute_texture(coef_dict_s['tex'])

        print("v_s shape", v_s.shape)
        print("tex_s shape", tex_s.shape)

        e_s = self.arcface(Xs)
        # r_s = self.hopenet(Xs)
        z_s = self.emodel(Xs)

        coeffs_d = self.face3d(Xd, compute_render=False)
        coef_dict_d = self.face3d.facemodel.split_coeff(coeffs_d)
        r_d = self.face3d.facemodel.compute_rotation(coef_dict_d['angle'])
        z_d = self.emodel(Xd)

        Y = self.decoder((v_s, e_s, None, z_s), (None, None, r_d, z_d))

        return Y



if __name__ == '__main__':

    model = Portrait()

    # Create an instance of the model

    # Create some dummy input data
    input_data = torch.randn(2, 3, 224, 224).to('cuda')
    input_data2 = torch.randn(2, 3, 224, 224).to('cuda')

    # Pass the input data through the model
    output = model(input_data, input_data2)

    # Print the shape of the output
    print(output.shape)
