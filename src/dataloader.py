# video_dataset.py

import os
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Lambda

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_files = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.mp4'):
                    self.video_files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        vr = VideoReader(video_path, ctx=cpu(0))
        video_data = vr.get_batch(range(len(vr))).asnumpy()  # Load entire video as a numpy array
        video_tensor = torch.tensor(video_data).float().to('cuda')  # Convert to tensor and transfer to GPU

        if self.transform:
            video_tensor = self.transform(video_tensor)  # Ensure transforms are compatible with CUDA tensors

        return video_tensor

# Define transformations
transform = Compose([
    Lambda(lambda x: x.permute(0, 3, 1, 2).float()),
    Lambda(lambda x: (x / 255.0) * 2 - 1)  # Normalize to [-1, 1]
])
