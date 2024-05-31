import torch
import torch.nn as nn
import torch.nn.functional as F
import dlib
import cv2

from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, arcface_model):
        super(PerceptualLoss, self).__init__()
        self.arcface = arcface_model
        self.landmarks_detector = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")


    def forward(self, pred, target):
        pred_features = self.arcface(pred)
        target_features = self.arcface(target)
        Lface = F.l1_loss(pred_features, target_features)



        return Lface 

class GazeLoss(nn.Module):
    def __init__(self, gaze_model):
        super(GazeLoss, self).__init__()
        self.gaze_model = gaze_model.eval()
        for param in self.gaze_model.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_gaze = self.gaze_model(pred)
        target_gaze = self.gaze_model(target)
        loss = F.l1_loss(pred_gaze, target_gaze)
        return loss

class GANLoss(nn.Module):
    def __init__(self, discriminator):
        super(GANLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, pred, target):
        real_loss = F.mse_loss(self.discriminator(target), torch.ones_like(target))
        fake_loss = F.mse_loss(self.discriminator(pred), torch.zeros_like(pred))
        loss = real_loss + fake_loss
        return loss

class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()

    def forward(self, z_src_to_drv, z_drv, z_src, z_drv_to_src):
        positive_pairs = [(z_src_to_drv, z_drv), (z_drv_to_src, z_src)]
        negative_pairs = [(z_src_to_drv, z_drv_to_src), (z_drv_to_src, z_src_to_drv)]
        
        positive_loss = sum(F.cosine_embedding_loss(z1, z2, torch.ones(z1.size(0)).to(z1.device)) for z1, z2 in positive_pairs)
        negative_loss = sum(F.cosine_embedding_loss(z1, z2, -torch.ones(z1.size(0)).to(z1.device)) for z1, z2 in negative_pairs)
        
        loss = positive_loss + negative_loss
        return loss

class CustomCriterion(nn.Module):
    def __init__(self, perceptual_weight=1.0, gaze_weight=1.0, gan_weight=1.0, cycle_weight=1.0):
        super(CustomCriterion, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.gaze_loss = GazeLoss(gaze_model=YourGazeModel())  # Replace with your gaze model
        self.gan_loss = GANLoss(discriminator=YourDiscriminator())  # Replace with your discriminator
        self.cycle_loss = CycleConsistencyLoss()
        
        self.perceptual_weight = perceptual_weight
        self.gaze_weight = gaze_weight
        self.gan_weight = gan_weight
        self.cycle_weight = cycle_weight

    def forward(self, preds, targets, z_src_to_drv, z_drv, z_src, z_drv_to_src):
        perceptual_loss = self.perceptual_loss(preds, targets)
        gaze_loss = self.gaze_loss(preds, targets)
        gan_loss = self.gan_loss(preds, targets)
        cycle_loss = self.cycle_loss(z_src_to_drv, z_drv, z_src, z_drv_to_src)
        
        total_loss = (self.perceptual_weight * perceptual_loss + 
                     self.gaze_weight * gaze_loss + 
                     self.gan_weight * gan_loss + 
                     self.cycle_weight * cycle_loss)
        return total_loss