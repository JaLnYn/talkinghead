from src.decoder.facedecoder import FaceDecoder
from src.encoder.emocoder import get_trainable_emonet
from src.encoder.deep3dfacerecon import get_face_recon_model
from src.encoder.hopenet import get_model_hopenet
from src.encoder.arcface import get_model_arcface
from src.encoder.eapp import get_eapp_model
import torch
import torch.nn as nn
import dlib

class Portrait(nn.Module):
    def __init__(self, eapp_path="./models/eapp_path", emo_path="./models/emo_path"):
        super().__init__()

        arcface_model_path = "./models/arcface2/model_ir_se50.pth"
        face3d_model_path = "./models/face3drecon.pth"
        hope_model_path = "./models/hopenet_robust_alpha1.pkl"

        self.detector = dlib.get_frontal_face_detector()

        self.face3d = get_face_recon_model(face3d_model_path)

        if eapp_path is not None:
            self.eapp = get_eapp_model(None, "cuda")
        else:
            self.eapp = get_eapp_model(None, "cuda")

        self.arcface = get_model_arcface(arcface_model_path)

        if emo_path is not None:
            self.emodel = get_trainable_emonet(emo_path)
        else:
            self.emodel = get_trainable_emonet(None)

        self.decoder = FaceDecoder()

    def forward(self, Xs, Xd):
        # input are images
        with torch.no_grad():

            coeffs_s = self.face3d(Xs, compute_render=False)
            coef_dict_s = self.face3d.facemodel.split_coeff(coeffs_s)
            r_s = coef_dict_s['angle']
            t_s = coef_dict_s['trans']
            e_s = self.arcface(Xs)

            coeffs_d = self.face3d(Xd, compute_render=False)
            coef_dict_d = self.face3d.facemodel.split_coeff(coeffs_d)
            r_d = coef_dict_d['angle']
            t_d = coef_dict_d['trans']
            e_d = self.arcface(Xd)
        
        v_s = self.eapp(Xs)
        z_s = self.emodel(Xs) # expression
        z_d = self.emodel(Xd)

        Y = self.decoder((v_s, e_s, r_s, t_s, z_s), (None, e_d, r_d, t_d, z_d))

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
