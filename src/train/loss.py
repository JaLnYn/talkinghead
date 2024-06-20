import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from src.train.discriminator import MultiScalePatchDiscriminator

class PerceptualLoss(nn.Module):
    def __init__(self, arcface_model, gaze_model, arcface_weight=0.1, imagenet_weight=0.1,  gaze_weight=0.1):
        super(PerceptualLoss, self).__init__()
        self.arcface_weight = arcface_weight
        self.imagenet_weight = imagenet_weight
        self.gaze_weight = gaze_weight
        self.arcface = arcface_model
        self.imageNet = resnet18(weights='IMAGENET1K_V1')

        # freeze image net
        for param in self.imageNet.parameters():
            param.requires_grad = False

        self.gaze_model = gaze_model

    def forward(self, source, driver, pred):
        batch_size = pred.size(0)

        # ArcFace loss
        pred_features = self.arcface(pred)
        target_features = self.arcface(source)
        Lface = F.cosine_embedding_loss(pred_features, target_features, torch.ones(pred_features.size(0)).to(pred_features.device))
        Lface_scaled = Lface * self.arcface_weight

        # ImageNet ResNet-18 loss
        pred_in = self.imageNet(pred)
        target_in = self.imageNet(source)
        Lin = torch.norm(pred_in - target_in, dim=1).mean()  # Normalize over batch
        Lin_scaled = Lin * self.imagenet_weight

        # Gaze loss
        gaze_pred_1 = self.gaze_model.get_gaze(pred)
        gaze_pred_2 = self.gaze_model.get_gaze(driver)
        Lgaze = torch.norm(gaze_pred_1 - gaze_pred_2, dim=1).mean()  # Normalize over batch
        Lgaze_scaled = Lgaze * self.gaze_weight

        # Calculate total weighted perceptual loss
        total_loss = Lface_scaled + Lin_scaled + Lgaze_scaled

        # Return individual losses along with the total
        return total_loss, {
            'Lface': Lface_scaled,
            'Lin': Lin_scaled,
            'Lgaze': Lgaze_scaled
        }


class GANLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(GANLoss, self).__init__()
        self.discriminator = MultiScalePatchDiscriminator(input_channels=3)
        self.weight = weight

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
                feature_matching_loss = feature_matching_loss + F.l1_loss(real_feat, fake_feat)

        # Normalize losses by number of scales and sum real and fake hinge losses
        total_loss = (real_loss + fake_loss) / len(real_outputs) + feature_matching_loss / sum(len(feats) for feats in real_features)
        return total_loss * self.weight
    
class CycleConsistencyLoss(nn.Module):
    def __init__(self,  emodel, weight=0.1, scale=5.0, margin=0.2):
        super(CycleConsistencyLoss, self).__init__()
        self.weight = weight
        self.emodel = emodel
        self.scale = scale
        self.margin = margin

    def forward(self, Xd, Xd_prime, gsd, gspd):
        batch_size = Xd.size(0)
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
        assert loss.size(0) == batch_size
        return sum(loss)*self.weight/batch_size



class VasaLoss(nn.Module):
    def __init__(self, face3d, arcface, emodel, gaze_model, gaze_weight=0.1, arcface_weight=0.1, emodel_weight=0.1, face3d_weight=0.1):
        super(VasaLoss, self).__init__()
        self.face3d = face3d
        self.arcface = arcface
        self.emodel = emodel
        self.gaze_model = gaze_model
        self.gaze_weight = gaze_weight
        self.arcface_weight = arcface_weight
        self.emodel_weight = emodel_weight
        self.face3d_weight = face3d_weight

    def forward(self, giiij, gjjij, gsd, gsmod):
        batch_size = giiij.size(0)
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

        cosloss = F.cosine_embedding_loss(zi, zj, torch.ones(zi.size(0)).to(zi.device))/batch_size * self.emodel_weight
        assert zi.size(0) == batch_size
        rotation_loss = sum(torch.norm(r_s - r_d, dim=1) + torch.norm(t_s - t_d, dim=1))/batch_size * self.face3d_weight
        assert r_s.size(0) == batch_size

        gaze_pred_1 = self.gaze_model.get_gaze(giiij)
        gaze_pred_2 = self.gaze_model.get_gaze(gjjij)
        assert gaze_pred_1.size(0) == batch_size
        gaze_loss = sum(torch.norm(gaze_pred_1 - gaze_pred_2, dim=1))/batch_size * self.gaze_weight

        esd = self.arcface(gsd)
        emod = self.arcface(gsmod)

        arcloss = F.cosine_embedding_loss(esd, emod, -torch.ones(esd.size(0)).to(esd.device))
        return (arcloss + gaze_loss + rotation_loss + cosloss, {"arcloss": arcloss, "gazeloss": gaze_loss, "rotationloss": rotation_loss, "cosloss": cosloss})


class PortraitLoss(nn.Module):
    def __init__(self, perceptual_weight=0.10, gaze_weight=0.10, gan_weight=1.0, cycle_weight=0.5, arcface_model=None, emodel=None, gaze_model=None):
        super(PortraitLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss(arcface_model, gaze_model)
        self.gan_loss = GANLoss()  # Replace with your discriminator
        self.cycle_loss = CycleConsistencyLoss(emodel)
        
        self.perceptual_weight = perceptual_weight
        self.gaze_weight = gaze_weight
        self.gan_weight = gan_weight
        self.cycle_weight = cycle_weight

    def forward(self, Xs, Xd, Xsp, Xdp, gsd, gspd): #(self, Xs, Xd, Xsp, Xdp, gsd, gsdp, gspd, gspdp):
        batch_size = Xs.size(0)
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

        return sum(self.perceptual_weight * Lper + self.cycle_weight * Lcyc)/batch_size + self.gan_weight * Lgan 
