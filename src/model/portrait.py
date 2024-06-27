import random
import os
import wandb

import tqdm

from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

from PIL import Image
import numpy as np

import torchvision.models as models
from src.model.generator import Generator

class Portrait(nn.Module):
    def __init__(self, config):
        super(Portrait, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.center_size = 224

        self.pose_encoder = models.resnet50()
        self.pose_encoder.fc = nn.Linear(2048, 512)

        self.iden_encoder = models.resnet50()
        self.iden_encoder.fc = nn.Linear(2048, 512)

        self.emot_encoder = models.resnet50()
        self.emot_encoder.fc = nn.Linear(2048, 512)

        self.resize = nn.Linear(1024, 512)

        self.generator = Generator(1024, 1024, 1024)

        self.to(self.device)


    def forward(self, Xs, Xd, zero_noise=False):
        batch_size = Xs.shape[0]
        Ep = self.pose_encoder(Xs)
        Ei = self.iden_encoder(Xs)
        Ee = self.emot_encoder(Xs)

        generator_input = self.resize(torch.cat([Ep,Ee], dim=1))
        Y = self.generator(torch.cat([Ei, generator_input], dim=1), 0.5, 6, zero_noise)

        start = (Y.shape[2] - self.center_size) // 2
        end = start + self.center_size

        # Perform center crop
        return Y[:, :, start:end, start:end]

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    model = Portrait(None)
    model.train()  # Ensure model is in training mode

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create some dummy input data
    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    input_data = video_dataset[0][0:2].to(model.device)
    input_data_backup = input_data.clone()  # Backup to check for modifications
    input_data_clone = input_data.clone()  # Clone to prevent modification
    input_data_clone.requires_grad = False

    # Forward pass to get initial outputs
    assert torch.allclose(input_data_clone, input_data_backup, atol=1e-6), "Input data differs"
    output = model(input_data_clone, input_data_clone, zero_noise=True)
    loss = output.mean()
    loss.backward()
    optimizer.step()  # Update weights with backpropagation

    # Get encoder outputs after training
    trained_pose, trained_iden, trained_emot = model.pose_encoder(input_data_clone), model.iden_encoder(input_data_clone), model.emot_encoder(input_data_clone)
    trained_output = model(input_data_clone, input_data_clone, zero_noise=True)

    # Save model
    saved_state_dict = model.state_dict()
    torch.save(model.state_dict(), 'portrait_model.pth')

    # Load model
    model_loaded = Portrait(None)

    # test unloaded  
    assert torch.allclose(input_data_clone, input_data_backup, atol=1e-6), "Input data differs"
    loaded_output = model_loaded(input_data_clone, input_data_clone, zero_noise=True)
    loaded_pose, loaded_iden, loaded_emot = model_loaded.pose_encoder(input_data_clone), model_loaded.iden_encoder(input_data_clone), model_loaded.emot_encoder(input_data_clone)
    assert not torch.allclose(loaded_output, trained_output, atol=1e-6), "Full model outputs same before load."
    assert not torch.allclose(trained_pose, loaded_pose, atol=1e-6), "Pose encoder outputs same before load."
    assert not torch.allclose(trained_iden, loaded_iden, atol=1e-6), "Identity encoder outputs same before load."
    assert not torch.allclose(trained_emot, loaded_emot, atol=1e-6), "Emotion encoder outputs same before load."
    del loaded_output, loaded_pose, loaded_iden, loaded_emot    


    loaded_state_dict = torch.load('portrait_model.pth')
    model_loaded.load_state_dict(loaded_state_dict)

    # noise = torch.randn(1, 1024).to('cuda')  # Replace z_dim with actual dimension
    # output1 = model.generator(noise, alpha=0.5, steps=1, zero_noise = True )  # Adjust parameters as necessary
    # output2 = model_loaded.generator(noise, alpha=0.5, steps=1, zero_noise = True)
    # assert torch.allclose(output1, output2, atol=1e-6), "Outputs are not close enough"


    saved_keys = set(saved_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())
    assert saved_keys == loaded_keys, "Mismatch in model state dict keys after loading."

    # Perform forward pass again with the loaded model
    assert torch.allclose(input_data_clone, input_data_backup, atol=1e-6), "Input data differs"
    loaded_output = model_loaded(input_data_clone, input_data_clone, zero_noise=True)
    loaded_pose, loaded_iden, loaded_emot = model_loaded.pose_encoder(input_data_clone), model_loaded.iden_encoder(input_data_clone), model_loaded.emot_encoder(input_data_clone)

    # Compare encoder outputs
    print(torch.mean(loaded_output - trained_output))
    print(torch.max(loaded_output - trained_output))
    print(torch.min(loaded_output - trained_output))
    assert torch.allclose(loaded_output, trained_output, atol=1e-6), "Full model outputs differ after load."
    assert torch.allclose(trained_pose, loaded_pose, atol=1e-6), "Pose encoder outputs differ after load."
    assert torch.allclose(trained_iden, loaded_iden, atol=1e-6), "Identity encoder outputs differ after load."
    assert torch.allclose(trained_emot, loaded_emot, atol=1e-6), "Emotion encoder outputs differ after load."

    print("All checks passed successfully. Encoder outputs are consistent after training and loading.")
