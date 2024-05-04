from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
from torch.optim import Adam

def main():
    # Setup the Dataset and DataLoader
    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    video_loader = DataLoader(video_dataset, batch_size=1, shuffle=True)

    # Run the test
    # Training loop with tqdm
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(enumerate(video_loader), total=len(video_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for i, video in progress_bar:
            # TODO
            # Forward pass
            # encoded_images = model(images)
            
            # # Compute loss (assuming some target is available)
            # # Here we use the encoded_images itself as a dummy target just for the sake of example.
            # # In practice, the target could be different aspects of the latent code.
            # loss = loss_function(encoded_images, encoded_images.detach())  

            # # Backward pass and optimize
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # total_loss += loss.item()

            # Update tqdm progress bar
            progress_bar.set_postfix({'loss': total_loss / (i + 1)})

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    # Save the model
    torch.save(model.state_dict(), 'encoder_model.pth') 

if __name__ == "__main__":
    main()
