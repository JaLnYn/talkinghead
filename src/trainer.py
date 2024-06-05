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
def load_data(root_dir, batch_size=8):
    dataset = VideoDataset(root_dir=root_dir, transform=transform)

    def collate_frames(batch):
        # Assume batch is a list of frames; shuffle to randomize frame selection across videos
        random.shuffle(batch) 

        # Assuming each batch now contains enough frames for two pairs
        half_point = len(batch) // 2

        Xs_stack = []
        Xd_stack = []
        for item in batch[:half_point]:
            Xs_stack.append(random.choice(item))
            Xd_stack.append(random.choice(item))
        Xs = torch.stack(Xs_stack)  
        Xd = torch.stack(Xd_stack)  

        
        Xs_prime_stack = []
        Xd_prime_stack = []
        for item in batch[half_point:]:
            Xs_prime_stack.append(random.choice(item))
            Xd_prime_stack.append(random.choice(item))
        Xs_prime = torch.stack(Xs_prime_stack)  
        Xd_prime = torch.stack(Xd_prime_stack)  

        return Xs, Xd, Xs_prime, Xd_prime

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_frames)
    return dataloader

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Script for handling emopath and eapp_path arguments.")
    
    parser.add_argument('--emopath', type=str, default=None, help='Path to the emotion model file. Default is None.')
    parser.add_argument('--eapp_path', type=str, default=None, help='Path to the application data file. Default is None.')
    parser.add_argument('--data_path', type=str, default='./dataset/mp4', help='Path to the dataset. The folder should be folder of mp4s.')
    parser.add_argument('--batch_size', type=int, default=16, help='batches')
    
    args = parser.parse_args()

    video_dataset = load_data(root_dir='./dataset/mp4', batch_size=args.batch_size)
    print(args.eapp_path, args.emopath)
    p = Portrait(args.eapp_path, args.emopath)
    p.train_model(video_dataset)


if __name__ == '__main__':
    main()
