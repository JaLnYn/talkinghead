import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from src.train.discriminator import MultiScalePatchDiscriminator

class PerceptualLoss(nn.Module):
    def __init__(self, arcface_model):
        super(PerceptualLoss, self).__init__()
        self.arcface = arcface_model
        # self.landmarks_detector = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
        self.imageNet = resnet18(weights='IMAGENET1K_V1')


    def forward(self, pred, source, driver):
        pred_features = self.arcface(pred)
        target_features = self.arcface(source)
        Lface = torch.norm(pred_features - target_features, dim=1)

        pred_in = self.arcface(pred)
        target_in = self.arcface(driver)
        Lin = torch.norm(pred_in - target_in, dim=1)


        # ADD GAZE LOSS

        return Lface + Lin

# class GazeLoss(nn.Module):
#     def __init__(self, gaze_model):
#         super(GazeLoss, self).__init__()
#         self.gaze_model = gaze_model.eval()
#         for param in self.gaze_model.parameters():
#             param.requires_grad = False
# 
#     def forward(self, pred, target):
#         pred_gaze = self.gaze_model(pred)
#         target_gaze = self.gaze_model(target)
#         loss = F.l1_loss(pred_gaze, target_gaze)
#         return loss

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
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()

    def forward(self, zsd, zd, zspd, zdp):
        positive_pairs = [(zsd, zd), (zspd, zd)]
        negative_pairs = [(zsd, zdp), (zspd, zdp)]
        
        positive_loss = sum(F.cosine_embedding_loss(z1, z2, torch.ones(z1.size(0)).to(z1.device)) for z1, z2 in positive_pairs)
        negative_loss = sum(F.cosine_embedding_loss(z1, z2, -torch.ones(z1.size(0)).to(z1.device)) for z1, z2 in negative_pairs)
        
        loss = positive_loss + negative_loss
        return loss

class PortraitLoss(nn.Module):
    def __init__(self, perceptual_weight=1.0, gaze_weight=1.0, gan_weight=1.0, cycle_weight=2.0, arcface_model=None):
        super(PortraitLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss(arcface_model)
        self.gan_loss = GANLoss()  # Replace with your discriminator
        self.cycle_loss = CycleConsistencyLoss()
        
        self.perceptual_weight = perceptual_weight
        self.gaze_weight = gaze_weight
        self.gan_weight = gan_weight
        self.cycle_weight = cycle_weight

    def forward(self, Xs, Xd, Xsd, Xsp, Xdp, Xspdp, zsd, zd, zspd, zdp):
        # Compute perceptual loss
        Lper = self.perceptual_loss(Xs, Xd, Xsd)
        Lper += self.perceptual_loss(Xsp, Xdp, Xspdp)

        Lgan = self.gan_loss(Xs, Xsd)
        Lgan += self.gan_loss(Xdp, Xspdp)

        Lcyc = self.cycle_loss(zsd, zd, zspd, zdp)

        return self.perceptual_weight * Lper + self.gan_weight * Lgan + self.cycle_weight * Lcyc
