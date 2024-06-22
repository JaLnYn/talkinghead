import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.mobilenetv2 import Conv2dNormActivation
from torchvision.transforms import Compose, Lambda

import numpy as np 

from PIL import Image
from torchvision import transforms

class GazeNet(nn.Module):

    def __init__(self, device):    
        super(GazeNet, self).__init__()
        self.device = device
        self.preprocess = transforms.Compose([
            # Lambda(lambda x: (x + 1)/2 * 255.0),  # Normalize to [-1, 1]
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            transforms.Resize((112,112)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model = torchvision.models.mobilenet_v2()
        model.features[-1] = Conv2dNormActivation(320, 256, kernel_size=1)
        self.backbone = model.features

        self.Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   
        self.fc_final = nn.Linear(512, 2)

        self._initialize_weight()
        self._initialize_bias()
        self.to(device)


    def _initialize_weight(self):
        nn.init.normal_(self.Conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Conv2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Conv3.weight, mean=0.0, std=0.001)

    def _initialize_bias(self):
        nn.init.constant_(self.Conv1.bias, val=0.1)
        nn.init.constant_(self.Conv2.bias, val=0.1)
        nn.init.constant_(self.Conv3.bias, val=1)

    def forward(self, x):
        # print(x.shape)
        x = self.backbone(x)
        # print(x.shape)
        y = F.relu(self.Conv1(x))
        y = F.relu(self.Conv2(y))
        y = F.relu(self.Conv3(y))
        
        x = F.dropout(F.relu(torch.mul(x, y)), 0.5)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        gaze = self.fc_final(x)

        return gaze

    def get_gaze(self, img):
        img = self.preprocess(img)
        x = self.forward(img.to(self.device))
        return x

def get_gaze_model():
    print('Loading MobileFaceGaze model...')
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = GazeNet(device)

    if(not torch.cuda.is_available()):
        print('Tried to load GPU but found none. Please check your environment')
    state_dict = torch.load("./models/gazenet.pth", map_location=device)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    print('Model loaded using {} as device'.format(device))
    return model


def reverse_transform():
    return Compose([
        Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),  # Denormalize
        #Lambda(lambda x: (x + 1) * 255.0/2),  # Denormalize
        #Lambda(lambda x: x[[2, 1, 0], :, :]),
        Lambda(lambda x: x.permute(1, 2, 0)),
    ])
    

if __name__ == "__main__":
    from src.dataloader import VideoDataset, transform
        
    model = get_gaze_model()
    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    input_data = video_dataset[0][0:3]
    input_data2 = video_dataset[10][0:3]
    print(input_data.shape)

    # Pass the input data through the model
    output = model.get_gaze(input_data)
    output2 = model.get_gaze(input_data2)

    # Print the shape of the output
    print(output)
    print(output2)
    from torchvision.transforms import ToPILImage
    rev_transform = reverse_transform()

    # Convert tensor to PIL Image and save
    to_pil = ToPILImage()
    import matplotlib.pyplot as plt
    for i, img_tensor in enumerate(input_data):
        plt.imshow(rev_transform(img_tensor).cpu())
        plt.axis('off')
        plt.savefig(f'test_{i+1}.png', bbox_inches='tight', pad_inches=0)

    for i, img_tensor in enumerate(input_data2):
        # Display the image using matplotlib
        plt.imshow(rev_transform(img_tensor).cpu())
        plt.axis('off')
        plt.savefig(f'test_{i+4}.png', bbox_inches='tight', pad_inches=0)



