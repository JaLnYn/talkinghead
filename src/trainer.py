from src.dataloader import VideoDataset
import yaml

from torch.utils.data import DataLoader
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from src.model import Portrait, train_model
import random


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
