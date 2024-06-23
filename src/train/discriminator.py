
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64, final_input_size=128):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters * 4)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filters * 8)
        self.final = nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        eat1 = self.leaky_relu(self.bn1(self.conv1(x)))
        feat2 = self.leaky_relu(self.bn2(self.conv2(feat1)))
        feat3 = self.leaky_relu(self.bn3(self.conv3(feat2)))
        feat4 = self.leaky_relu(self.bn4(self.conv4(feat3)))
        out = self.final(feat4)
        # Return both the final output and the list of feature maps
        return out, [feat1, feat2, feat3, feat4]


class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64):
        super(MultiScalePatchDiscriminator, self).__init__()
        self.scale1_discriminator = PatchDiscriminator(input_channels, num_filters, 676)
        self.scale2_discriminator = PatchDiscriminator(input_channels, num_filters, 144)
        self.scale3_discriminator = PatchDiscriminator(input_channels, num_filters, 25)
        
    def forward(self, x):
        # Downsample images for different scales
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

        # Process each scale and collect outputs and features
        out1, feats1 = self.scale1_discriminator(x)
        out2, feats2 = self.scale2_discriminator(x2)
        out3, feats3 = self.scale3_discriminator(x3)

        # Aggregate outputs and features from all scales
        outputs = [out1, out2, out3]
        features = [feats1, feats2, feats3]
        return outputs, features

