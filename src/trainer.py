from src.dataloader import VideoDataset

from torch.utils.data import DataLoader
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from src.model import Portrait
import random



# def select_random_frames(dataloader, batch_size, device):
#     video_dataset = dataloader.dataset
#     batch_Xs, batch_Xd, batch_Xs_prime, batch_Xd_prime = [], [], [], []
#     
#     for _ in range(batch_size):
#         idx1, idx2 = random.sample(range(len(video_dataset)), 2)
#         video1 = video_dataset[idx1]
#         video2 = video_dataset[idx2]
#         
#         video1_idx1, video1_idx2 = random.sample(range(len(video1)), 2)
#         video2_idx1, video2_idx2 = random.sample(range(len(video2)), 2)
#         
#         Xs = video1[video1_idx1]
#         Xd = video1[video1_idx2]
#         Xs_prime = video2[video2_idx1]
#         Xd_prime = video2[video2_idx2]
# 
#         print(Xs.shape, Xd.shape, Xs_prime.shape, Xd_prime.shape)
#         exit()
#         
#         batch_Xs.append(Xs)
#         batch_Xd.append(Xd)
#         batch_Xs_prime.append(Xs_prime)
#         batch_Xd_prime.append(Xd_prime)
#     
#     return torch.cat(batch_Xs), torch.cat(batch_Xd), torch.cat(batch_Xs_prime), torch.cat(batch_Xd_prime)

# Main function to execute the pipeline

    # Load and preprocess the data
def load_data(root_dir, batch_size=4):
    dataset = VideoDataset(root_dir=root_dir, transform=transform)

    def collate_frames(batch):
        # Assume batch is a list of frames; shuffle to randomize frame selection across videos
        random.shuffle(batch) 

        # Assuming each batch now contains enough frames for two pairs
        quarter_point = len(batch) // 4
        Xs = torch.stack([item[0] for item in batch[:quarter_point]])  # First quarter for Xs
        Xd = torch.stack([item[0] for item in batch[quarter_point:2*quarter_point]])  # Second quarter for Xd
        Xs_prime = torch.stack([item[0] for item in batch[2*quarter_point:3*quarter_point]])  # Third quarter for Xs_prime
        Xd_prime = torch.stack([item[0] for item in batch[3*quarter_point:]])  # Last quarter for Xd_prime

        return Xs, Xd, Xs_prime, Xd_prime

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_frames)
    return dataloader


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Script for handling emopath and eapp_path arguments.")
    
    parser.add_argument('--emopath', type=str, default=None, help='Path to the emotion model file. Default is None.')
    parser.add_argument('--eapp_path', type=str, default=None, help='Path to the application data file. Default is None.')
    parser.add_argument('--data_path', type=str, default='./dataset/mp4', help='Path to the dataset. The folder should be folder of mp4s.')
    parser.add_argument('--batch_size', type=int, default=12, help='batches')
    
    args = parser.parse_args()

    video_dataset = load_data(root_dir='./dataset/mp4')
    print(args.eapp_path, args.emopath)
    p = Portrait(args.eapp_path, args.emopath)
    p.train_model(video_dataset)
    
    batch_size = 24


if __name__ == '__main__':
    main()