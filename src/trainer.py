from src.dataloader import VideoDataset
import yaml

from torch.utils.data import DataLoader
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from src.model.loss import PerceptualLoss, GANLoss, CycleConsistencyLoss
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
import tqdm
import os

from src.model.portrait import Portrait


def collate_frames(batch):
    """
    Custom collate function that processes a batch of tensors, each containing sampled frames from videos.

    Args:
    batch (list of Tensors): The batch containing video frame tensors.

    Returns:
    tuple: A tuple containing processed tensors ready for model input.
    """
    # Randomly select two frames from each half of the batch for processing
    Xs_stack = []
    Xd_stack = []
    Xs_prime_stack = []
    Xd_prime_stack = []

    half_point = len(batch) // 2
    for item in batch[:half_point]:
        if item.shape[0] > 1:  # Ensure there is more than one frame
            indices = random.sample(range(item.shape[0]), 2)
            Xs_stack.append(item[indices[0]])
            Xd_stack.append(item[indices[1]])

    for item in batch[half_point:]:
        if item.shape[0] > 1:
            indices = random.sample(range(item.shape[0]), 2)
            Xs_prime_stack.append(item[indices[0]])
            Xd_prime_stack.append(item[indices[1]])

    # Stack all selected frames to create batches
    if Xs_stack and Xd_stack and Xs_prime_stack and Xd_prime_stack:  # Check if lists are not empty
        Xs = torch.stack(Xs_stack)
        Xd = torch.stack(Xd_stack)
        Xs_prime = torch.stack(Xs_prime_stack)
        Xd_prime = torch.stack(Xd_prime_stack)
        
        # Concatenate Xs with Xs_prime and Xd with Xd_prime
        Xs_combined = torch.cat((Xs, Xs_prime), dim=0)
        Xd_combined = torch.cat((Xd, Xd_prime), dim=0)
        Xs_prime_combined = torch.cat((Xs_prime, Xs), dim=0)
        Xd_prime_combined = torch.cat((Xd_prime, Xd), dim=0)

        return Xs_combined, Xd_combined, Xs_prime_combined, Xd_prime_combined
    else:
        # Return zero tensors if not enough frames were available
        zero_tensor = torch.zeros((1, 3, 224, 224))  # Adjust dimensions as per your model's requirement
        return zero_tensor, zero_tensor, zero_tensor, zero_tensor

def load_data(root_dir, batch_size=8, transform=None):
    """
    Function to load data using VideoDataset and DataLoader with a custom collate function.
    
    Args:
    root_dir (str): Path to the directory containing videos.
    batch_size (int): Number of samples per batch.
    transform (callable, optional): Transformations applied to video frames.

    Returns:
    DataLoader: A DataLoader object ready for iteration.
    """
    dataset = VideoDataset(root_dir=root_dir, transform=transform, frames_per_clip=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_frames)
    return dataloader


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

    perceptual_loss = PerceptualLoss(config)
    gan_loss = GANLoss(config, discriminator=p.discriminator)

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0

        # Wrap the training loader with tqdm for a progress bar
        train_iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_loader))
        log_interval = 10  # len(train_loader) // 20
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

            Eis, Ees, Eps = p.encode(Xs)
            # Epd, Eid, Eed = p.encode(Xd, return_components=True)
            gsd = p.decode(Eis, Ees, Eps, zero_noise=False)

            Lper = perceptual_loss(Xd, gsd)
            Lgan = gan_loss(Xd, gsd)

            # Lvasa = self.v1loss(giiij, gjjij, gsd, gsmod)

            total_loss = Lper[0] + Lgan[0] 

            running_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                wandb.log({
                    'Example Source': wandb.Image(Xs[0].cpu().detach().numpy().transpose(1, 2, 0)),
                    'Example Driver': wandb.Image(Xd[0].cpu().detach().numpy().transpose(1, 2, 0)),
                    'Example Output': wandb.Image(gsd[0].cpu().detach().numpy().transpose(1, 2, 0)),
                })

            wandb_log = {
                'Epoch': epoch + 1,
                'Total Loss': total_loss.item()
            }

            # if self.config['weights']['perceptual']['gaze'] != 0:
            #     wandb_log['Gaze Loss'] = Lper[1]['Lgaze'].item()
            if p.config['weights']['perceptual']['lpips'] != 0:
                wandb_log['lpips Loss'] = Lper[1]['lpips'].item()
            if p.config['weights']['gan']['real'] + p.config['weights']['gan']['fake'] + p.config['weights']['gan'][
                'feature_matching'] != 0:
                wandb_log['GAN Loss'] = Lgan[0].item()
            if p.config['weights']['gan']['real'] != 0:
                wandb_log['GAN real Loss'] = Lgan[1]['real_loss'].item()
            if p.config['weights']['gan']['fake'] != 0:
                wandb_log['GAN fake Loss'] = Lgan[1]['fake_loss'].item()
            if p.config['weights']['gan']['adversarial'] != 0:
                wandb_log['GAN adversarial Loss'] = Lgan[1]['adversarial_loss'].item()
            if p.config['weights']['gan']['feature_matching'] != 0:
                wandb_log['Gan feature Loss'] = Lgan[1]['feature_matching_loss'].item()

            wandb.log(wandb_log)

            train_iterator.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss.item():.4f}"
            )

            step += 1

        # p.save_model(path="./models/portrait/epoch{}/".format(epoch))
        p.save_model(path=f"./models/portrait/{p.config['training']['name']}/epoch{epoch}/", epoch=epoch,
                     optimizer=optimizer)
        print(f'Epoch {epoch + 1}, Average Loss {running_loss / len(train_loader):.4f}')

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
        
    video_dataset = load_data(root_dir='./dataset/mp4', transform=transform, batch_size=config["training"]["batch_size"])
    p = Portrait(config)

    train_model(config, p, video_dataset)


if __name__ == '__main__':
    main()
