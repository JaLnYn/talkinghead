import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from src.train.discriminator import MultiScalePatchDiscriminator

class PerceptualLoss(nn.Module):
    def __init__(self, arcface_model, gaze_model):
        super(PerceptualLoss, self).__init__()
        self.arcface = arcface_model
        self.imageNet = resnet18(weights='IMAGENET1K_V1')
        self.gaze_model = gaze_model

    def forward(self, pred, source, driver):
        pred_features = self.arcface(pred)
        target_features = self.arcface(source)

        Lface = F.cosine_embedding_loss(pred_features, target_features, torch.ones(pred_features.size(0)).to(pred_features.device))

        pred_in = self.imageNet(pred) # image net
        target_in = self.imageNet(source)
        Lin = torch.norm(pred_in - target_in, dim=1)

        # ADD GAZE LOSS
        # TODO gaze discrepancy here
        
        gaze_pred_1 = self.gaze_model.get_gaze(pred)
        gaze_pred_2 = self.gaze_model.get_gaze(driver)
        Lgaze = torch.norm(gaze_pred_1 - gaze_pred_2, dim=1)
        return Lface + Lin + Lgaze


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
            real_loss = real_loss  + torch.mean(F.relu(1.0 - real_output))
            fake_loss = fake_loss  + torch.mean(F.relu(1.0 + fake_output))

        # Compute feature matching loss
        feature_matching_loss = 0
        # Iterate over each scale
        for scale_real_feats, scale_fake_feats in zip(real_features, fake_features):
            # For each scale, iterate over each feature map
            for real_feat, fake_feat in zip(scale_real_feats, scale_fake_feats):
                # Calculate L1 loss between corresponding features from real and fake images
                feature_matching_loss = feature_matching_loss + F.l1_loss(real_feat.detach(), fake_feat)

        # Normalize losses by number of scales and sum real and fake hinge losses
        total_loss = (real_loss + fake_loss) / len(real_outputs) + feature_matching_loss / sum(len(feats) for feats in real_features)
        return total_loss

class CycleConsistencyLoss(nn.Module):
    def __init__(self, emodel, scale=5.0, margin=0.2):
        super(CycleConsistencyLoss, self).__init__()
        self.emodel = emodel
        self.scale = scale
        self.margin = margin

    def forward(self, Xd, Xd_prime, gsd, gspd):
        zd = self.emodel(Xd)
        zdp = self.emodel(Xd_prime)
        zsd = self.emodel(gsd)
        zspd = self.emodel(gspd)



        # Calculate cosine similarity and apply margin and scale
        def cosine_distance(z1, z2):
            cosine_similarity = F.cosine_similarity(z1, z2)
            return torch.exp(self.scale * (cosine_similarity - self.margin))

         # Calculate distances

        # Define the positive and negative pairs
        negative_pairs = [(zsd, zdp), (zspd, zdp)]
        neg_distances = torch.sum(torch.exp(torch.stack([cosine_distance(z1, z2) for z1, z2 in negative_pairs])))

        loss = - torch.log( cosine_distance(zsd, zd) / (cosine_distance(zsd, zd) + neg_distances + 1e-6))
        loss = loss - torch.log( cosine_distance(zspd, zd) / (cosine_distance(zspd, zd) + neg_distances + 1e-6))

        return loss

class VasaLoss(nn.Module):
    def __init__(self, face3d, arcface, emodel, gaze_model):
        super(VasaLoss, self).__init__()
        self.face3d = face3d
        self.arcface = arcface
        self.emodel = emodel
        self.gaze_model = gaze_model

    def forward(self, giiij, gjjij, gsd, gsmod):
        # Compute perceptual loss
        zi = self.emodel(giiij)
        zj = self.emodel(gjjij)

        with torch.no_grad():

            coeffs_s = self.face3d(giiij, compute_render=False)
            coef_dict_s = self.face3d.facemodel.split_coeff(coeffs_s)
            r_s = coef_dict_s['angle']
            t_s = coef_dict_s['trans']

            coeffs_d = self.face3d(gjjij, compute_render=False)
            coef_dict_d = self.face3d.facemodel.split_coeff(coeffs_d)
            r_d = coef_dict_d['angle']
            t_d = coef_dict_d['trans']

        loss = F.cosine_embedding_loss(zi, zj, torch.ones(zi.size(0)).to(zi.device))
        loss = loss + torch.norm(r_s - r_d, dim=1) + torch.norm(t_s - t_d, dim=1)
        # TODO gaze discrepancy here

        gaze_pred_1 = self.gaze_model.get_gaze(giiij)
        gaze_pred_2 = self.gaze_model.get_gaze(gjjij)
        loss = loss + torch.norm(gaze_pred_1 - gaze_pred_2, dim=1)

        esd = self.arcface(gsd)
        emod = self.arcface(gsmod)

        loss = loss + F.cosine_embedding_loss(esd, emod, torch.ones(esd.size(0)).to(esd.device))
        return loss


class PortraitLoss(nn.Module):
    def __init__(self, perceptual_weight=1.0, gaze_weight=1.0, gan_weight=1.0, cycle_weight=2.0, arcface_model=None, emodel=None, gaze_model=None):
        super(PortraitLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss(arcface_model, gaze_model)
        self.gan_loss = GANLoss()  # Replace with your discriminator
        self.cycle_loss = CycleConsistencyLoss(emodel)
        
        self.perceptual_weight = perceptual_weight
        self.gaze_weight = gaze_weight
        self.gan_weight = gan_weight
        self.cycle_weight = cycle_weight

    def forward(self, Xs, Xd, Xsp, Xdp, gsd, gspd): #(self, Xs, Xd, Xsp, Xdp, gsd, gsdp, gspd, gspdp):
        # Compute perceptual loss
        Lper = self.perceptual_loss(Xs, Xd, gsd)
        # Lper = Lper + self.perceptual_loss(Xs, Xdp, gsdp)
        Lper = Lper + self.perceptual_loss(Xsp, Xd, gspd)
        # Lper = Lper + self.perceptual_loss(Xsp, Xdp, gspdp)

        Lgan = self.gan_loss(Xs, gsd)
        # Lgan = Lgan + self.gan_loss(Xs, gsdp)
        Lgan = Lgan + self.gan_loss(Xsp, gspd)
        # Lgan = Lgan + self.gan_loss(Xsp, gspdp)

        Lcyc = self.cycle_loss(Xd, Xdp, gsd, gspd)

        return sum(self.perceptual_weight * Lper + self.gan_weight * Lgan + self.cycle_weight * Lcyc)
