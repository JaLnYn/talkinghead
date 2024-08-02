# video_dataset.py

import os
import torch
import random
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Lambda, Normalize

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.video_files = [os.path.join(subdir, file)
                            for subdir, dirs, files in os.walk(root_dir)
                            for file in files if file.endswith('.mp4')]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = random.sample(range(total_frames), self.frames_per_clip)
            video_data = vr.get_batch(frame_indices).asnumpy()
            video_tensor = torch.from_numpy(video_data).float().to('cuda')

            if self.transform:
                video_tensor = self.transform(video_tensor)

            return video_tensor
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

# Define transformations
transform = Compose([
    Lambda(lambda x: x.permute(0, 3, 1, 2).float()),
    Lambda(lambda x: (x / 255.0)),  # Normalize to [0, 1]
])

