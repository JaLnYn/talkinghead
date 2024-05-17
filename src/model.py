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
        self.hopenet = get_model_hopenet(hope_model_path)
        self.arcface = get_model_arcface(arcface_model_path)
        self.emodel = get_trainable_emonet(emo_model_path)

        self.decoder = FaceDecoder()

    def forward(self, Xs, Xd):
        # input are images
        v_s = self.face3d(Xs)
        e_s = self.arcface(Xs)
        r_s = self.hopenet(Xs)
        z_s = self.emodel(Xs)

        r_d = self.hopenet(Xd)
        z_d = self.emodel(Xd)

        Y = self.decoder((v_s, e_s, r_s, z_s), (None, None, r_d, z_d))

        return Y



if __name__ == '__main__':

    model = Portrait()

    # Create an instance of the model

    # Create some dummy input data
    input_data = torch.randn(3, 3, 224, 224).to('cuda')
    input_data2 = torch.randn(3, 3, 224, 224).to('cuda')

    # Pass the input data through the model
    output = model(input_data, input_data2)

    # Print the shape of the output
    print(output.shape)
