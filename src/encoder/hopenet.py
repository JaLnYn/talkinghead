import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
import copy 
from torchvision import transforms

class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)
        self.idx_tensor = torch.FloatTensor(list(range(num_bins))).unsqueeze(0).cuda()
        self.n_bins = num_bins
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        real_yaw = self.fc_yaw(x)
        real_pitch = self.fc_pitch(x)
        real_roll = self.fc_roll(x)
        real_yaw = torch.softmax(real_yaw, dim=1)
        real_pitch = torch.softmax(real_pitch, dim=1)
        real_roll = torch.softmax(real_roll, dim=1)
        real_yaw = (real_yaw * self.idx_tensor).sum(dim=1)
        real_pitch = (real_pitch * self.idx_tensor).sum(dim=1)
        real_roll = (real_roll * self.idx_tensor).sum(dim=1)
        real_yaw = (real_yaw - self.n_bins // 2) * 3 * np.pi / 180
        real_pitch = (real_pitch - self.n_bins // 2) * 3 * np.pi / 180
        real_roll = (real_roll - self.n_bins // 2) * 3 * np.pi / 180

        return real_yaw, real_pitch, real_roll, x

def get_model_hopenet(pretrained_path):
    pretrained_hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=66).cuda()
    pretrained_hopenet.load_state_dict(torch.load(pretrained_path, map_location=torch.device("cuda")))
    pretrained_hopenet.requires_grad_(False)
    return pretrained_hopenet


transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == "__main__":
    from src.encoder.vggface import vgg_face_dag
    from torch.utils.data import DataLoader
    from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
    import matplotlib.pyplot as plt
    import torch
    import numpy as np


    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)

    # https://github.com/natanielruiz/deep-head-pose/blob/master/code/test_on_video.py
    print(video_dataset[0].shape)
    print(video_dataset[0][0].shape)
    pretrained_path = "./models/hopenet_robust_alpha1.pkl" # https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR
    pretrained_hopenet = get_model_hopenet(pretrained_path)
    frame = video_dataset[0][0].reshape(1, 224, 224, 3 ).permute(0, 3, 1, 2)
    trans = transformations(frame).cuda()
    print(pretrained_hopenet(trans))
