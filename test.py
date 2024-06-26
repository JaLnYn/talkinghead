import torch
import lpips
from PIL import Image
import numpy as np
from torchvision import transforms

# Load the LPIPS model
lpips_model = lpips.LPIPS(net='vgg').to("cuda")  # Using AlexNet as the backbone
# for param in lpips_model.parameters():
#     param.requires_grad = False 

# Function to load and preprocess an image
def load_image(img_path):
    image = Image.open(img_path).convert('RGB')

    image = np.array(image)
    image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0  # Convert to tensor and normalize
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image.requires_grad = False
    image_min = torch.min(image)
    image_max = torch.max(image)
    print("Min value of image:", image_min)
    print("Max value of image:", image_max)
    return image.to("cuda")

# Paths to your images
image_paths = [f'test_{i}.png' for i in range(1, 7)]

# Load images
images = torch.stack([load_image(path) for path in image_paths])  # Batch the images
images2 = images.clone()  # Clone the images

distances = lpips_model(images, images2)  # Calculate the distances between the first image and the rest


print("Distances within the same batch:")
print(distances)

# Function to shift images in the batch
def shift_images(image_batch):
    return torch.roll(image_batch, shifts=-1, dims=0)

# Shift images and compare
shifted_images = shift_images(images)
distances_shifted = lpips_model(images, shifted_images)

print("Distances between original and shifted batch:")
print(distances_shifted)

optimizer = torch.optim.Adam([images], lr=0.01)

import tqdm

t = tqdm.tqdm(range(1000))
for i in t:

    optimizer.zero_grad()
    distances_shifted = lpips_model(images, shifted_images).mean()
    distances_shifted.backward()
    t.set_description(f"Loss: {distances_shifted.item()}")
    optimizer.step()
