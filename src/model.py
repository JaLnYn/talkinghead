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

        self.vggface = None #TODO

        self.eapp = Eapp ()

        self.emodel = Emocoder()

        self.decoder = FaceDecoder()

        self.discriminator = MultiScalePatchDiscriminator(3)

        # self.gaze_model = get_gaze_model()
        self.vggface = InceptionResnetV1(pretrained='vggface2').to(self.device)

        # self.v1loss = VasaLoss(config, face3d=self.face3d, vggface=self.arcface, emodel=self.emodel, gaze_model=self.gaze_model)
        

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
        torch.save(self.state_dict(), path + "portrait.pth")
        torch.save(model_state, os.path.join(path, "checkpoint.pth"))
        

    def load_model(self, path="./models/portrait/"):
        self.load_state_dict(torch.load(path + "portrait.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f'Model loaded from {path}')
        
def train_model(config, p, train_loader):
    optimizer = torch.optim.Adam(p.parameters(), lr=p.config["training"]["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p.to(device)
    wandb.init(project='portrait_project', resume="allow")

    checkpoint_path = f"./models/portrait/{p.config['training']['name']}/"
    num_epochs = p.config["training"]["num_epochs"]

    start_epoch = 0

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) == 0:
            latest_epoch = max([int(epoch_dir.split("epoch")[1]) for epoch_dir in os.listdir(checkpoint_path)])
            checkpoint_path = os.path.join(checkpoint_path, f"epoch{latest_epoch}/checkpoint.pth")

            checkpoint = torch.load(checkpoint_path)
            p.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch

    perceptual_loss = PerceptualLoss(config, vggface=p.vggface)
    gan_loss = GANLoss(config, discriminator=p.discriminator)
    cycle_loss = CycleConsistencyLoss(config, emodel=p.emodel)

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0

        # Wrap the training loader with tqdm for a progress bar
        train_iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_loader))
        log_interval = 10 # len(train_loader) // 20
        step = 0
        for Xs, Xd, Xsp, Xdp in train_iterator:
            min_batch_size = min(Xs.size(0), Xd.size(0), Xsp.size(0), Xdp.size(0))

            # Check if the minimum batch size is zero
            if min_batch_size == 0:
                continue  # Skip this iteration

            # Crop batches to the minimum batch size
            Xs = Xs[:min_batch_size].to(device)
            Xd = Xd[:min_batch_size].to(device)
            Xsp = Xsp[:min_batch_size].to(device)
            Xdp = Xdp[:min_batch_size].to(device)

            optimizer.zero_grad()

            gsd, (v_s, e_s, r_s, t_s, z_s), (v_d, e_d, r_d, t_d, z_d)  = p(Xs, Xd, return_components=True)
            gspd, (v_sp, e_sp, r_sp, t_sp, z_sp), (v_d1, e_d1, r_d1, t_d1, z_d1) = p(Xsp, Xd, return_components=True)

            # construct vasa loss
            # giiij = p.decoder((v_s, e_s, r_s, t_s, z_s), (None, None, r_s, t_s, z_d))
            # gjjij = p.decoder((v_d, e_d, r_d, t_d, z_d), (None, None, r_s, t_s, z_d))
            # gsmod = p.decoder((v_sp, e_sp, r_sp, t_sp, z_sp), (None, None, r_d, t_d, z_d))

            # loss = p.loss(Xs, Xd, Xsp, Xdp, gsd, gspd)

            Lper = perceptual_loss(Xs, Xd, gsd)
            Lgan = gan_loss(Xd, gsd)
            Lcyc = cycle_loss(Xd, Xdp, gsd, gspd)

            # Lvasa = self.v1loss(giiij, gjjij, gsd, gsmod)

            total_loss = Lper[0] + Lcyc + Lgan[0]# + Lvasa[0]

            running_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()
            
            if step % log_interval == 0:
                wandb.log({
                    'Example Source': wandb.Image(Xs[0].cpu().detach().numpy().transpose(1, 2, 0)),
                    'Example Source Prime': wandb.Image(Xsp[0].cpu().detach().numpy().transpose(1, 2, 0)),
                    'Example Driver': wandb.Image(Xd[0].cpu().detach().numpy().transpose(1, 2, 0)),
                    'Example Output': wandb.Image(gsd[0].cpu().detach().numpy().transpose(1, 2, 0)),
                    'Example Output SPD': wandb.Image(gspd[0].cpu().detach().numpy().transpose(1, 2, 0)),
                })
            

            # if step % 5 == 0:
            #    if step % 20 == 0:
            #        print(f"Step {step}, {gsd[0].cpu().detach().numpy().transpose(1, 2, 0)}")
            #    Image.fromarray(np.uint8((Xs[0]*256).cpu().detach().numpy().transpose(1, 2, 0))).save(f"./test/Xs{step}.png")
            #    Image.fromarray(np.uint8((gsd[0]*256).cpu().detach().numpy().transpose(1, 2, 0))).save(f"./test/Xsd{step}.png")

            wandb_log = {
                'Epoch': epoch + 1,
                'Total Loss': total_loss.item()
            }

            # if self.config['weights']['perceptual']['gaze'] != 0:
            #     wandb_log['Gaze Loss'] = Lper[1]['Lgaze'].item()
            if p.config['weights']['perceptual']['imagenet'] != 0:
                wandb_log['ImageNet Loss'] = Lper[1]['Lin'].item()
            if p.config['weights']['perceptual']['arcface'] != 0:
                wandb_log['Face Loss'] = Lper[1]['Lface'].item()
            if p.config['weights']['perceptual']['vggface'] != 0:
                wandb_log['VggFace Loss'] = Lper[1]['vggface'].item()
            if p.config['weights']['perceptual']['lpips'] != 0:
                wandb_log['lpips Loss'] = Lper[1]['lpips'].item()
            if p.config['weights']['gan']['real'] + p.config['weights']['gan']['fake'] + p.config['weights']['gan']['feature_matching']!= 0:
                wandb_log['GAN Loss'] = Lgan[0].item()
            if p.config['weights']['gan']['real'] != 0:
                wandb_log['GAN real Loss'] = Lgan[1]['real_loss'].item()
            if p.config['weights']['gan']['fake'] != 0:
                wandb_log['GAN fake Loss'] = Lgan[1]['fake_loss'].item()
            if p.config['weights']['gan']['adversarial'] != 0:
                wandb_log['GAN adversarial Loss'] = Lgan[1]['adversarial_loss'].item()
            if p.config['weights']['gan']['feature_matching'] != 0:
                wandb_log['Gan feature Loss'] = Lgan[1]['feature_matching_loss'].item()
            if p.config['weights']['cycle'] != 0:
                wandb_log['Cycle Loss'] = Lcyc.item()
            # if p.config['weights']['vasa']['arcface'] != 0:
            #     wandb_log['ArcFace Loss'] = Lvasa[1]['arcloss'].item()
            # if p.config['weights']['vasa']['face3d'] != 0:
            #     wandb_log['Face3D Loss'] = Lvasa[1]['rotationloss'].item()
            # if p.config['weights']['vasa']['emodel'] != 0:
            #     wandb_log['EModel Loss'] = Lvasa[1]['cosloss'].item()
            # if p.config['weights']['vasa']['gaze'] != 0:
            #     wandb_log['Second Gaze Loss'] = Lvasa[1]['gazeloss'].item()

            wandb.log(wandb_log)

            train_iterator.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss.item():.4f}"
            )

            step += 1
        
        # p.save_model(path="./models/portrait/epoch{}/".format(epoch))
        p.save_model(path=f"./models/portrait/{p.config['training']['name']}/epoch{epoch}/", epoch=epoch, optimizer=optimizer)
        print(f'Epoch {epoch+1}, Average Loss {running_loss / len(train_loader):.4f}')

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
