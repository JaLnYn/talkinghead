import random
import os
import wandb

import tqdm

from src.decoder.facedecoder import FaceDecoder
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.utils.data import DataLoader
from src.encoder.emocoder import get_trainable_emonet
from src.encoder.deep3dfacerecon import get_face_recon_model
from src.encoder.arcface import get_model_arcface
from src.encoder.eapp import get_eapp_model
from src.train.loss import PerceptualLoss, GANLoss, CycleConsistencyLoss, VasaLoss
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

        # self.loss = PortraitLoss(emodel=self.emodel, arcface_model=self.arcface, gaze_model=self.gaze_model)

        self.v1loss = VasaLoss(face3d=self.face3d, arcface=self.arcface, emodel=self.emodel, gaze_model=self.gaze_model)
        self.perceptual_loss = PerceptualLoss(arcface_model=self.arcface, gaze_model=self.gaze_model)
        self.gan_loss = GANLoss(weight=1.0)
        self.cycle_loss = CycleConsistencyLoss(weight=1.0, emodel=self.emodel)

        self.perceptual_weight = 1.0
        self.gan_weight = 1.0
        self.cycle_weight = 2.0

    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, start_epoch=0, checkpoint_path=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        wandb.init(project='portrait_project', resume="allow")
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch

        for epoch in range(start_epoch, num_epochs):
            running_loss = 0
            running_gaze_loss = 0
            running_image_net_loss = 0
            running_face_loss = 0
            running_gan_loss = 0
            running_cycle_loss = 0
            running_arcface_loss = 0
            running_emodel_loss = 0
            running_gaze_loss2 = 0
            running_face3d_loss = 0

            # Wrap the training loader with tqdm for a progress bar
            train_iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_loader))
            for Xs, Xd, Xsp, Xdp in train_iterator:
                min_batch_size = min(Xs.size(0), Xd.size(0), Xsp.size(0), Xdp.size(0))
                print(Xs.size(0), Xd.size(0), Xsp.size(0), Xdp.size(0))

                # Check if the minimum batch size is zero
                if min_batch_size == 0:
                    continue  # Skip this iteration

                # Crop batches to the minimum batch size
                Xs = Xs[:min_batch_size].to(device)
                Xd = Xd[:min_batch_size].to(device)
                Xsp = Xsp[:min_batch_size].to(device)
                Xdp = Xdp[:min_batch_size].to(device)

                optimizer.zero_grad()

                gsd, (v_s, e_s, r_s, t_s, z_s), (v_d, e_d, r_d, t_d, z_d)  = self(Xs, Xd, return_components=True)
                gspd, (v_sp, e_sp, r_sp, t_sp, z_sp), (v_d, e_d, r_d, t_d, z_d) = self(Xsp, Xd, return_components=True)

                # construct vasa loss
                giiij = self.decoder((v_s, e_s, r_s, t_s, z_s), (None, None, r_s, t_s, z_d))
                gjjij = self.decoder((v_d, e_d, r_d, t_d, z_d), (None, None, r_s, t_s, z_d))
                gsmod = self.decoder((v_sp, e_sp, r_sp, t_sp, z_sp), (None, None, r_d, t_d, z_d))

                # loss = self.loss(Xs, Xd, Xsp, Xdp, gsd, gspd)

                Lper = self.perceptual_loss(Xs, Xd, gsd) + self.perceptual_loss(Xsp, Xd, gspd)
                Lgan = self.gan_loss(Xs, gsd) + self.gan_loss(Xsp, gspd)
                Lcyc = self.cycle_loss(Xd, Xdp, gsd, gspd)

                Lvasa = self.v1loss(giiij, gjjij, gsd, gsmod)

                total_loss = Lper[0] + Lcyc + Lgan + Lvasa[0]

                running_loss += total_loss.item()
                running_gaze_loss += Lper[1]['Lgaze']
                running_image_net_loss += Lper[1]['Lin']
                running_face_loss += Lper[1]['Lface']
                running_gan_loss += Lgan.item()
                running_cycle_loss += Lcyc.item()
                running_arcface_loss += Lvasa[1]['arcloss']
                running_face3d_loss += Lvasa[1]['rotationloss']
                running_emodel_loss += Lvasa[1]['cosloss']
                running_gaze_loss2 += Lvasa[1]['gazeloss']


                total_loss.backward()
                optimizer.step()

                wandb.log({
                    'Total Loss': total_loss.item(),
                    'Gaze Loss': Lper[1]['Lgaze'],
                    'ImageNet Loss': Lper[1]['Lin'],
                    'Face Loss': Lper[1]['Lface'],
                    'GAN Loss': Lgan.item(),
                    'Cycle Loss': Lcyc.item(),
                    'ArcFace Loss': Lvasa[1]['arcloss'],
                    'Face3D Loss': Lvasa[1]['rotationloss'],
                    'EModel Loss': Lvasa[1]['cosloss'],
                    'Second Gaze Loss': Lvasa[1]['gazeloss']
                })

                train_iterator.set_description(
                    f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss.item():.4f}, "
                    f"Face: {running_face_loss:.4f}, INet: {running_image_net_loss:.4f}, "
                    f"Gaze: {running_gaze_loss:.4f}, Gaze2: {running_gaze_loss2:.4f}, "
                    f"Gan: {running_gan_loss:.4f}, Cycle: {running_cycle_loss:.4f}, "
                    f"ArcFace: {running_arcface_loss:.4f}, EModel: {running_emodel_loss:.4f}, "
                    f"Face3D: {running_face3d_loss:.4f}"
                )
            
            # self.save_model(path="./models/portrait/epoch{}/".format(epoch))
            self.save_model(path="./models/portrait/epoch{}/".format(epoch), epoch=epoch, optimizer=optimizer)
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

    def save_model(self, path, epoch, optimizer):
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(self.state_dict(), path + "portrait.pth")
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
