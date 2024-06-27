import random
import os
import wandb

import tqdm

from src.decoder.facedecoder import FaceDecoder
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.utils.data import DataLoader
from src.encoder.emocoder import Emocoder
from src.encoder.deep3dfacerecon import get_face_recon_model
from src.encoder.arcface import get_model_arcface
from src.encoder.eapp import Eapp
from src.train.loss import PerceptualLoss, GANLoss, CycleConsistencyLoss
from src.train.discriminator import MultiScalePatchDiscriminator
from src.train.eye_tracking import get_gaze_model
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

from PIL import Image
import numpy as np

class Portrait(nn.Module):
    def __init__(self, config):
        super(Portrait, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.pose_encoder =



    def forward(self, Xs, Xd, return_components=False):
        # input are images

        # coeffs_s = self.face3d(Xs, compute_render=False)
        # coef_dict_s = self.face3d.facemodel.split_coeff(coeffs_s)
        # r_s = coef_dict_s['angle']
        # t_s = coef_dict_s['trans']
        e_s = None # self.arcface(Xs)
        r_s, t_s, r_d, t_d = None, None, None, None

        # coeffs_d = self.face3d(Xd, compute_render=False)
        # coef_dict_d = self.face3d.facemodel.split_coeff(coeffs_d)
        # r_d = coef_dict_d['angle']
        # t_d = coef_dict_d['trans']
        
        v_s = self.eapp(Xs)
        z_s = self.emodel(Xs) # expression
        z_d = self.emodel(Xd)

        Y = self.decoder((v_s, e_s, r_s, t_s, z_s), (None, None, r_d, t_d, z_d))
        if return_components:
            v_d = self.eapp(Xd)
            e_d = None # self.arcface(Xd)
            return Y, (v_s, e_s, r_s, t_s, z_s), (v_d, e_d, r_d, t_d, z_d)
        return Y

    def save_model(self, path, epoch, optimizer):
        if not os.path.exists(path):
            os.makedirs(path)
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(self.state_dict(), os.path.join("portrait.pth"))
        torch.save(model_state, os.path.join(path, "checkpoint.pth"))
        

    def load_model(self, path="./models/portrait/"):
        self.load_state_dict(torch.load(path + "portrait.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f'Model loaded from {path}')
        


if __name__ == '__main__':

    model = Portrait(None, None)

    # Create an instance of the model

    # Create some dummy input data
    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    input_data = video_dataset[0][0:3]
    input_data2 = video_dataset[0][1:4]
    print(input_data.shape)

    # Pass the input data through the model
    output = model(input_data, input_data2)

    # Print the shape of the output
    print(output.shape)
