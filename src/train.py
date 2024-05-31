from src.dataloader import VideoDataset

from torch.utils.data import DataLoader
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.model import Portrait
import random




# Load and preprocess the data
def load_data(root_dir, batch_size=4, transform=None):
    dataset = VideoDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def select_random_frames(dataloader, device):
    video_dataset = dataloader.dataset
    idx1, idx2 = random.sample(range(len(video_dataset)), 2)
    video1 = video_dataset[idx1]
    video2 = video_dataset[idx2]
    
    frame1_idx1, frame2_idx1 = random.sample(range(len(video1)), 2)
    frame1_idx2, frame2_idx2 = random.sample(range(len(video2)), 2)
    
    Xs = video1[frame1_idx1].unsqueeze(0).permute(0, 3, 1, 2).to(device)
    Xd = video2[frame1_idx1].unsqueeze(0).permute(0, 3, 1, 2).to(device)
    Xs_prime = video1[frame2_idx1].unsqueeze(0).permute(0, 3, 1, 2).to(device)
    Xd_prime = video2[frame2_idx2].unsqueeze(0).permute(0, 3, 1, 2).to(device)

    print(Xs.shape, Xd.shape, Xs_prime.shape, Xd_prime.shape)
    
    return Xs, Xd, Xs_prime, Xd_prime


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

# Train the model
def train_model(dataloader, model, criterion, optimizer, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            inputs = data.to(device)
            labels = torch.randint(0, 10, (inputs.size(0),)).to(device)  # Dummy labels, replace with actual labels
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:  # Print every 10 mini-batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Main function to execute the pipeline
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Script for handling emopath and eapp_path arguments.")
    
    parser.add_argument('--emopath', type=str, default=None, help='Path to the emotion model file. Default is None.')
    parser.add_argument('--eapp_path', type=str, default=None, help='Path to the application data file. Default is None.')
    parser.add_argument('--data_path', type=str, default='./dataset/mp4', help='Path to the dataset. The folder should be folder of mp4s.')
    parser.add_argument('--batch_size', type=int, default=12, help='batches')
    
    args = parser.parse_args()

    
    
    batch_size = 24
    dataloader = load_data(args.data_path, batch_size, transform)
    
    model = Portrait(args.emopath, args.eapp_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(dataloader, model, criterion, optimizer, num_epochs=5)
    
    print(f'Model Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()