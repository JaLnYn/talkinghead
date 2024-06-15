import random
import os

import tqdm

from src.decoder.facedecoder import FaceDecoder
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.utils.data import DataLoader
from src.encoder.emocoder import get_trainable_emonet
from src.encoder.deep3dfacerecon import get_face_recon_model
from src.encoder.arcface import get_model_arcface
from src.encoder.eapp import get_eapp_model
from src.train.loss import PortraitLoss, VasaLoss
from src.train.eye_tracking import get_gaze_model
import torch
import torch.nn as nn
import dlib

class Portrait(nn.Module):
    def __init__(self, eapp_path="./models/eapp.pth", emo_path="./models/emodel.pth"):
        super(Portrait, self).__init__()


        arcface_model_path = "./models/arcface2/model_ir_se50.pth"
        face3d_model_path = "./models/face3drecon.pth"

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

        self.gaze_model = get_gaze_model()

        self.loss = PortraitLoss(emodel=self.emodel, arcface_model=self.arcface, gaze_model=self.gaze_model)
        self.v1loss = VasaLoss(face3d=self.face3d, arcface=self.arcface, emodel=self.emodel, gaze_model=self.gaze_model)


    def select_frames(self, video, device):
        # Randomly select two different frames
        frame_indices = random.sample(range(len(video)), 2)
        return video[frame_indices[0]].unsqueeze(0).to(device), video[frame_indices[1]].unsqueeze(0).to(device)


    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, start_epoch=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        for epoch in range(start_epoch, num_epochs):
            running_loss = 0
            # Wrap the training loader with tqdm for a progress bar
            train_iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_loader))
            for Xs, Xd, Xsp, Xdp in train_iterator:
                Xs, Xd, Xsp, Xdp = Xs.to(device), Xd.to(device), Xsp.to(device), Xdp.to(device)
                optimizer.zero_grad()

                gsd, (v_s, e_s, r_s, t_s, z_s), (v_d, e_d, r_d, t_d, z_d)  = self(Xs, Xd, return_components=True)
                gspd, (v_sp, e_sp, r_sp, t_sp, z_sp), (v_d, e_d, r_d, t_d, z_d) = self(Xsp, Xd, return_components=True)

                # construct vasa loss
                giiij = self.decoder((v_s, e_s, r_s, t_s, z_s), (None, None, r_s, t_s, z_d))
                gjjij = self.decoder((v_d, e_d, r_d, t_d, z_d), (None, None, r_s, t_s, z_d))
                gsmod = self.decoder((v_sp, e_sp, r_sp, t_sp, z_sp), (None, None, r_d, t_d, z_d))

                loss = self.loss(Xs, Xd, Xsp, Xdp, gsd, gspd)
                loss = loss + self.v1loss(giiij, gjjij, gsd, gsmod)


                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_iterator.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss {loss.item():.4f}")
            print(f'Epoch {epoch+1}, Average Loss {running_loss / len(train_loader):.4f}')




    def forward(self, Xs, Xd, return_components=False):
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
        
        v_s = self.eapp(Xs)
        z_s = self.emodel(Xs) # expression
        z_d = self.emodel(Xd)

        Y = self.decoder((v_s, e_s, r_s, t_s, z_s), (None, None, r_d, t_d, z_d))
        if return_components:
            v_d = self.eapp(Xd)
            e_d = self.arcface(Xd)
            return Y, (v_s, e_s, r_s, t_s, z_s), (v_d, e_d, r_d, t_d, z_d)
        return Y

    def save_model(self, path="./models/portrait/"):
        os.makedirs(path, exist_ok=True)
        self.eapp.save_model(path + "eapp.pth")
        self.emodel.save_model(path + "emodel.pth")

        self.decoder.save_model(path + "decoder/")

    def load_model(self, path="./models/portrait/"):
        self.eapp.load_model(path + "eapp.pth")
        self.emodel.load_model(path + "emodel.pth")

        self.decoder.load_model(path + "decoder/")

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
