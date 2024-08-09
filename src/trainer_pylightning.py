import yaml
import random
import math
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import tqdm

from src.dataloader import VideoDataset, transform
from src.model.loss import PerceptualLoss, GANLoss, CycleConsistencyLoss, IEPLoss
from src.model.portrait import Portrait


def collate_frames(batch):
    # Same collate function as before
    Xs_stack = []
    Xd_stack = []
    Xs_prime_stack = []
    Xd_prime_stack = []

    half_point = len(batch) // 2
    for item in batch[:half_point]:
        if item is None:
            continue
        if item.shape[0] > 1:
            indices = random.sample(range(item.shape[0]), 2)
            Xs_stack.append(item[indices[0]])
            Xd_stack.append(item[indices[1]])

    for item in batch[half_point:]:
        if item is None:
            continue
        if item.shape[0] > 1:
            indices = random.sample(range(item.shape[0]), 2)
            Xs_prime_stack.append(item[indices[0]])
            Xd_prime_stack.append(item[indices[1]])

    if Xs_stack and Xd_stack and Xs_prime_stack and Xd_prime_stack:
        Xs = torch.stack(Xs_stack)
        Xd = torch.stack(Xd_stack)
        Xs_prime = torch.stack(Xs_prime_stack)
        Xd_prime = torch.stack(Xd_prime_stack)

        Xs_combined = torch.cat((Xs, Xs_prime), dim=0)
        Xd_combined = torch.cat((Xd, Xd_prime), dim=0)
        Xs_prime_combined = torch.cat((Xs_prime, Xs), dim=0)
        Xd_prime_combined = torch.cat((Xd_prime, Xd), dim=0)

        return Xs_combined, Xd_combined, Xs_prime_combined, Xd_prime_combined
    else:
        zero_tensor = torch.zeros((1, 3, 224, 224))
        return zero_tensor, zero_tensor, zero_tensor, zero_tensor


def load_data(root_dir, batch_size=8, transform=None):
    dataset = VideoDataset(root_dir=root_dir, transform=transform, frames_per_clip=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_frames)
    return dataloader

## checkpoint

class PortraitTrainer(pl.LightningModule):
    def __init__(self, config):
        super(PortraitTrainer, self).__init__()
        self.config = config
        self.p = Portrait(config)
        self.perceptual_loss = PerceptualLoss(config)
        self.gan_loss = GANLoss(config, model=self.p)
        self.iep_loss = IEPLoss(config, model=self.p)
        self.initial_resolution = config["training"]["initial_resolution"]
        self.final_resolution = config["training"]["final_resolution"]
        self.epochs_per_stage = config["training"]["epochs_per_stage"]
        self.transition_epochs = config["training"]["transition_epochs"]
        self.epochs_per_full_stage = self.epochs_per_stage + self.transition_epochs

    def forward(self, x):
        return self.p(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.p.parameters(), lr=self.config["training"]["learning_rate"])
        return optimizer

    def training_step(self, batch, batch_idx):
        Xs, Xd, Xsp, Xdp = batch
        epoch = self.current_epoch

        min_batch_size = min(Xs.size(0), Xd.size(0), Xsp.size(0), Xdp.size(0))
        if min_batch_size == 0:
            return None

        Xs = Xs[:min_batch_size]
        Xd = Xd[:min_batch_size]
        Xsp = Xsp[:min_batch_size]
        Xdp = Xdp[:min_batch_size]

        current_resolution = min(self.initial_resolution * 2 ** (epoch // self.epochs_per_full_stage), self.final_resolution)
        step = int(math.log2(current_resolution)) - 2
        in_transition = (epoch % self.epochs_per_full_stage) >= self.epochs_per_stage

        if not in_transition:
            passed_transitions = 0
        else:
            passed_transitions = epoch % self.epochs_per_full_stage - self.epochs_per_stage

        if in_transition:
            alpha = (passed_transitions * len(self.train_dataloader()) + batch_idx) / (self.transition_epochs * len(self.train_dataloader()))
            alpha = max(0, min(alpha, 1))
        else:
            alpha = 0

        Eid, Eed, Epd = self.p.encode(Xd)
        Eis, Ees, Eps = self.p.encode(Xs)

        gd = self.p.decode(Eid, Eed, Epd, alpha, step)
        gs = self.p.decode(Eis, Ees, Eps, alpha, step)

        gid = self.p.decode(Eis, Eed, Epd, alpha, step)
        gis = self.p.decode(Eid, Ees, Eps, alpha, step)

        ged = self.p.decode(Eid, Ees, Epd, alpha, step)
        ges = self.p.decode(Eis, Eed, Eps, alpha, step)

        gpd = self.p.decode(Eid, Eed, Eps, alpha, step)
        gps = self.p.decode(Eis, Ees, Epd, alpha, step)

        Liep = self.iep_loss(gs, gd, gis, gid, ges, ged, gps, gpd)
        Lper = self.perceptual_loss(Xd, gd)
        Lgan = self.gan_loss(Xd, gd, alpha, step)

        total_loss = Lper[0] + Lgan[0] + Liep[0]

        self.log('total_loss', total_loss)
        self.log('Lper', Lper[0])
        self.log('Lgan', Lgan[0])
        self.log('Liep', Liep[0])

        if batch_idx % 100 == 0 and self.config["training"]["use_wandb"]:
            wandb.log({
                'Example Source': wandb.Image(Xs[0].cpu().detach().numpy().transpose(1, 2, 0)),
                'Example Driver': wandb.Image(Xd[0].cpu().detach().numpy().transpose(1, 2, 0)),
                'Example Output': wandb.Image(gd[0].cpu().detach().numpy().transpose(1, 2, 0)),
            })

        wandb_log = {
            'Epoch': epoch + 1,
            'Total Loss': total_loss.item(),
            'Alpha': alpha,
            'Step': step
        }

        if self.config['weights']['perceptual']['vgg'] != 0:
            wandb_log['vgg loss'] = Lper[1]['vgg'].item()
        if self.config['weights']['perceptual']['lpips'] != 0:
            wandb_log['lpips Loss'] = Lper[1]['lpips'].item()
        if self.config['weights']['irfd']['i'] != 0:
            wandb_log['Identity IRFD Loss'] = Liep[1]['iden_loss'].item()
        if self.config['weights']['irfd']['e'] != 0:
            wandb_log['Emotion IRFD Loss'] = Liep[1]['emot_loss'].item()
        if self.config['weights']['irfd']['p'] != 0:
            wandb_log['Pose IRFD Loss'] = Liep[1]['pose_loss'].item()
        if self.config['weights']['gan']['real'] + self.config['weights']['gan']['fake'] + self.config['weights']['gan']['feature_matching'] != 0:
            wandb_log['GAN Loss'] = Lgan[0].item()
        if self.config['weights']['gan']['real'] != 0:
            wandb_log['GAN real Loss'] = Lgan[1]['real_loss'].item()
        if self.config['weights']['gan']['fake'] != 0:
            wandb_log['GAN fake Loss'] = Lgan[1]['fake_loss'].item()
        if self.config['weights']['gan']['adversarial'] != 0:
            wandb_log['GAN adversarial Loss'] = Lgan[1]['adversarial_loss'].item()

        
        # if p.config['weights']['gan']['feature_matching'] != 0:
        #     wandb_log['Gan feature Loss'] = Lgan[1]['feature_matching_loss'].item()

        if self.config["training"]["use_wandb"]:
            wandb.log(wandb_log)

        return total_loss

    def train_dataloader(self):
        return load_data(root_dir=self.config["training"]["data_path"], transform=transform, batch_size=self.config["training"]["batch_size"])

    def on_train_epoch_end(self):
        checkpoint_path = f"./models/portrait/{self.config['training']['name']}/epoch{self.current_epoch}/"
        self.p.save_model(path=checkpoint_path, epoch=self.current_epoch, optimizer=self.optimizers(), current_resolution=self.initial_resolution)
        print(f'Epoch {self.current_epoch + 1}, Average Loss: {self.trainer.callback_metrics["total_loss"].item():.4f}')

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Script for handling emopath and eapp_path arguments.")
    parser.add_argument('--config_path', type=str, default=None, help='Path to the config')
    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        print("Config path is None")
        assert False

    if config["training"]["use_wandb"]:
        wandb.init(project='portrait_project', resume="allow", config=config)

    latest_checkpoint = find_latest_checkpoint(config["training"]["model_path"])

    trainer = pl.Trainer(default_root_dir=config["training"]["model_path"], max_epochs=config["training"]["num_epochs"], devices=-1 if torch.cuda.is_available() else 0, accelerator="gpu" if torch.cuda.is_available() else None, strategy='ddp_find_unused_parameters_true'
)

    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model = PortraitTrainer.load_from_checkpoint(latest_checkpoint)
    else:
        print("initializing model")
        model = PortraitTrainer(config)

    trainer.fit(model)


if __name__ == '__main__':
    main()

