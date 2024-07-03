import random
import os
import wandb

import tqdm

from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.utils.data import DataLoader

import torch
import torch.nn as nn



import torchvision.models as models
from src.model.generator import Generator, Discriminator
# from src.model.discriminator import MultiScalePatchDiscriminator

class Portrait(nn.Module):
    def __init__(self, config):
        super(Portrait, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.center_size = 224

        self.discriminator = Discriminator(512)

        self.pose_encoder = models.resnet50()
        self.pose_encoder.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Tanh()
        )

        self.iden_encoder = models.resnet50()
        self.iden_encoder.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Tanh()
        )

        self.emot_encoder = models.resnet50()
        self.emot_encoder.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Tanh()
        )

        self.generator = Generator(1536, 512)

        self.to(self.device)


    def encode(self, X):
        Ei = self.iden_encoder(X)
        Ep = self.pose_encoder(X)
        Ee = self.emot_encoder(X)
        return Ei, Ep, Ee

    def decode(self, Ei, Ep, Ee, alpha, step, zero_noise=False):
        Y = self.generator(torch.cat([Ei, Ep, Ee], dim=1), alpha, step, zero_noise)
        return Y

    def discriminator_forward(self, X, alpha, step):
        return self.discriminator(X, alpha, step)

    def forward(self, Xs, Xd, alpha, step, zero_noise=False):
        Eis = self.iden_encoder(Xs)
        Epd = self.pose_encoder(Xd)
        Eed = self.emot_encoder(Xd)

        Y = self.generator(torch.cat([Eis, Epd, Eed], dim=1), alpha, step, zero_noise)

        return Y
    
    def save_model(self, path, epoch, optimizer, current_resolution):
        if not os.path.exists(path):
            os.makedirs(path)
        model_state = {
            'epoch': epoch,
            'current_resolution':current_resolution ,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(self.state_dict(), path + "portrait.pth")
        torch.save(model_state, os.path.join(path, "checkpoint.pth"))

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    # Load the YAML file

    import yaml

    with open('config/local_train.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Use the loaded config to initialize the model
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
    output = model(input_data_clone, input_data_clone, 0.5, 6, zero_noise=True)
    print(max(output.flatten()), min(output.flatten()))
    discrim_out = model.discriminator_forward(output, 0.5, 6)
    loss = output.mean() + discrim_out[0][0].mean() # random losss doesnt matter
    loss.backward()
    optimizer.step()  # Update weights with backpropagation

    # Get encoder outputs after training
    trained_pose, trained_iden, trained_emot = model.pose_encoder(input_data_clone), model.iden_encoder(input_data_clone), model.emot_encoder(input_data_clone)
    trained_output = model(input_data_clone, input_data_clone, 0.5, 6,  zero_noise=True)
    trained_discrim_out = model.discriminator_forward(trained_output, 0.5, 6)

    # Save model
    saved_state_dict = model.state_dict()
    torch.save(model.state_dict(), 'portrait_model.pth')

    # Load model
    model_loaded = Portrait(None)

    # test unloaded  
    assert torch.allclose(input_data_clone, input_data_backup, atol=1e-6), "Input data differs"
    loaded_output = model_loaded(input_data_clone, input_data_clone, 0.5,6,  zero_noise=True)
    discrim_out_loaded = model_loaded.discriminator_forward(output, 0.5, 6)
    loaded_pose, loaded_iden, loaded_emot = model_loaded.pose_encoder(input_data_clone), model_loaded.iden_encoder(input_data_clone), model_loaded.emot_encoder(input_data_clone)
    assert not torch.allclose(trained_discrim_out, discrim_out_loaded, atol=1e-6), "Full model outputs same before load."
    assert not torch.allclose(trained_discrim_out, discrim_out_loaded, atol=1e-6), "Full model outputs same before load."
    assert not torch.allclose(loaded_output, trained_output, atol=1e-6), "Full model outputs same before load."
    assert not torch.allclose(trained_pose, loaded_pose, atol=1e-6), "Pose encoder outputs same before load."
    assert not torch.allclose(trained_iden, loaded_iden, atol=1e-6), "Identity encoder outputs same before load."
    assert not torch.allclose(trained_emot, loaded_emot, atol=1e-6), "Emotion encoder outputs same before load."
    del loaded_output, loaded_pose, loaded_iden, loaded_emot, discrim_out_loaded


    loaded_state_dict = torch.load('portrait_model.pth')
    model_loaded.load_state_dict(loaded_state_dict)

    saved_keys = set(saved_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())
    assert saved_keys == loaded_keys, "Mismatch in model state dict keys after loading."

    # Perform forward pass again with the loaded model
    assert torch.allclose(input_data_clone, input_data_backup, atol=1e-6), "Input data differs"
    loaded_output = model_loaded(input_data_clone, input_data_clone, 0.5,6, zero_noise=True)
    loaded_pose, loaded_iden, loaded_emot = model_loaded.pose_encoder(input_data_clone), model_loaded.iden_encoder(input_data_clone), model_loaded.emot_encoder(input_data_clone)
    discrim_out_loaded = model_loaded.discriminator_forward(loaded_output, 0.5, 6)

    # Compare encoder outputs
    assert torch.allclose(trained_discrim_out, discrim_out_loaded, atol=1e-6), "Full model outputs differ after load."
    assert torch.allclose(trained_discrim_out, discrim_out_loaded, atol=1e-6), "Full model outputs differ after load."
    assert torch.allclose(loaded_output, trained_output, atol=1e-6), "Full model outputs differ after load."
    assert torch.allclose(trained_pose, loaded_pose, atol=1e-6), "Pose encoder outputs differ after load."
    assert torch.allclose(trained_iden, loaded_iden, atol=1e-6), "Identity encoder outputs differ after load."
    assert torch.allclose(trained_emot, loaded_emot, atol=1e-6), "Emotion encoder outputs differ after load."
    
    del trained_output, loaded_pose, loaded_iden, loaded_emot, discrim_out_loaded, loaded_output

    #### TESTING LOSSES
    from loss import PerceptualLoss, GANLoss

    target_data = torch.randn_like(output)
    target_data = torch.nn.Parameter(target_data)

    # Add target_data to the optimizer
    optimizer.add_param_group({'params': target_data})

    # Function to test loss
    def test_loss(loss_fn, t_data):
        input_data_clone = input_data.clone().to(model.device)
        
        # Compute the loss
        target_data = t_data.clone().to(model.device)
        loss = loss_fn(output, target_data)

        # Backward pass with retain_graph=True
        loss[0].backward(retain_graph=True)

        # Check if the gradients are consistent
        assert torch.allclose(input_data_clone.grad, input_data_clone.grad.clone()), "Gradients differ for the same input"

        # Update weights with optimizer
        optimizer.step()

        # Compute the loss again
        loss_after_backward = loss_fn(output, target_data)

        # Check if the loss remains the same after backward pass
        assert torch.allclose(loss, loss_after_backward), "Loss changed after backward pass"

    # Test with different loss functions
    test_loss(PerceptualLoss(config), target_data)
    test_loss(GANLoss(config), target_data)

    print("All checks passed successfully. Encoder outputs are consistent after training and loading.")
