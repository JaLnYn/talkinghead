from .decoder.facedecoder import FaceDecoder
from .encoder.emocoder import get_trainable_emonet
from .encoder.deep3dfacerecon import get_face_recon_model
from .encoder.hopenet import get_model_hopenet
from .encoder.arcface import get_model_arcface
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(MyModel, self).__init__()

        arcface_model_path = "./models/arcface2/model_ir_se50.pth"
        3dface_model_path = "./models/face3drecon.pth"
        hope_model_path = "./models/hopenet_robust_alpha1.pkl"
        emo_model_path = "./models/emo_path"

        self.3dface = get_face_recon_model(3dface_model_path)
        self.hopenet = get_model_hopenet(hope_model_path)
        self.arcface = get_model_arcface(arcface_model_path)
        self.emodel = get_trainable_emonet(emo_model_path)

        self.decoder = FaceDecoder

    def forward(self, Xs, Xd):
        # input are images
        v_s = self.3dface(Xs)
        e_s = self.arcnet(Xs)
        r_s = self.hopenet(Xs)
        z_s = self.emodel(Xs)

        r_d = self.hopenet(Xd)
        z_d = self.emodel(Xd)

        Y = self.decoder((v_s, e_s, r_s, z_s), (None, None, r_d, z_d))

        return Y



if __name__ == '__main__':
    # Create an instance of the encoder
    encoder = FaceEncoder()

    # Create an instance of the decoder
    decoder = FaceDecoder()

    # Create an instance of the model
    model = MyModel(encoder, decoder)

    # Create some dummy input data
    input_data = torch.randn(1, 3, 256, 256)

    # Pass the input data through the model
    output = model(input_data)

    # Print the shape of the output
    print(output.shape)
