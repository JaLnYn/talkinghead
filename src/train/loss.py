import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from src.train.discriminator import MultiScalePatchDiscriminator

class PerceptualLoss(nn.Module):
    def __init__(self, arcface_model):
        super(PerceptualLoss, self).__init__()
        self.arcface = arcface_model
        self.imageNet = resnet18(weights='IMAGENET1K_V1')

    def forward(self, pred, source, driver):
        pred_features = self.arcface(pred)
        target_features = self.arcface(source)
        Lface = torch.norm(pred_features - target_features, dim=1)

        pred_in = self.imageNet(pred) # image net
        target_in = self.imageNet(source)
        Lin = torch.norm(pred_in - target_in, dim=1)

        # ADD GAZE LOSS
        return Lface + Lin

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.discriminator = MultiScalePatchDiscriminator(input_channels=3)

    def forward(self, real, fake):
        # Get discriminator outputs and features for both real and fake images
        real_outputs, real_features = self.discriminator(real)
        fake_outputs, fake_features = self.discriminator(fake)

        # Compute hinge loss for real and fake images
        real_loss = 0
        fake_loss = 0
        for real_output, fake_output in zip(real_outputs, fake_outputs):
            real_loss += torch.mean(F.relu(1.0 - real_output))
            fake_loss += torch.mean(F.relu(1.0 + fake_output))

        # Compute feature matching loss
        feature_matching_loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            for rf, ff in zip(real_feat, fake_feat):
                feature_matching_loss += F.l1_loss(rf.detach(), ff)

        # Combine hinge and feature matching losses
        loss = (real_loss + fake_loss) / len(real_outputs) + feature_matching_loss / len(real_features[0])
        return loss

class CycleConsistencyLoss(nn.Module):
    def __init__(self, emodel):
        super(CycleConsistencyLoss, self).__init__()
        self.emodel = emodel 


    def forward(self, Xd, Xd_prime, gsd, gspd):

        zd = self.emodel(Xd)
        zdp = self.emodel(Xd_prime)

        zsd = self.emodel(gsd)
        zspd = self.emodel(gspd)


        positive_pairs = [(zsd, zd), (zspd, zd)]
        negative_pairs = [(zsd, zdp), (zspd, zdp)]
        
        positive_loss = sum(F.cosine_embedding_loss(z1, z2, torch.ones(z1.size(0)).to(z1.device)) for z1, z2 in positive_pairs)
        negative_loss = sum(F.cosine_embedding_loss(z1, z2, -torch.ones(z1.size(0)).to(z1.device)) for z1, z2 in negative_pairs)
        
        loss = positive_loss + negative_loss
        return loss

class PortraitLoss(nn.Module):
    def __init__(self, perceptual_weight=1.0, gaze_weight=1.0, gan_weight=1.0, cycle_weight=2.0, arcface_model=None, emodel=None):
        super(PortraitLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss(arcface_model)
        self.gan_loss = GANLoss()  # Replace with your discriminator
        self.cycle_loss = CycleConsistencyLoss(emodel)
        
        self.perceptual_weight = perceptual_weight
        self.gaze_weight = gaze_weight
        self.gan_weight = gan_weight
        self.cycle_weight = cycle_weight

    def forward(self, Xs, Xd, Xsp, Xdp, gsd, gsdp, gspd, gspdp):
        # Compute perceptual loss
        Lper = self.perceptual_loss(Xs, Xd, gsd)
        Lper += self.perceptual_loss(Xs, Xdp, gsdp)
        Lper += self.perceptual_loss(Xsp, Xd, gspd)
        Lper += self.perceptual_loss(Xsp, Xdp, gspdp)

        Lgan = self.gan_loss(Xs, gsd)
        Lgan += self.gan_loss(Xs, gsdp)
        Lgan += self.gan_loss(Xsp, gspd)
        Lgan += self.gan_loss(Xsp, gspdp)

        Lcyc = self.cycle_loss(Xd, Xdp, gsd, gspd)

        return self.perceptual_weight * Lper + self.gan_weight * Lgan + self.cycle_weight * Lcyc
