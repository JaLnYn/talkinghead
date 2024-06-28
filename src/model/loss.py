import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from facenet_pytorch import InceptionResnetV1
from src.model.discriminator import MultiScalePatchDiscriminator
from torchvision.transforms import Normalize

import lpips

class PerceptualLoss(nn.Module):
    def __init__(self, config):
        super(PerceptualLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.lpips = lpips.LPIPS(net='vgg').to(self.device)

        self.lpips_weight = config["weights"]["perceptual"]["lpips"]


    def forward(self, driver, pred):

        lpips_loss = self.lpips(self.normalize(pred), self.normalize(driver)).mean() * self.lpips_weight

        # Return individual losses along with the total
        total_loss = lpips_loss
        return total_loss, {
            'lpips': lpips_loss,
        }


class GANLoss(nn.Module):
    def __init__(self, config):
        super(GANLoss, self).__init__()
        self.real_weight = config["weights"]["gan"]["real"] 
        self.fake_weight = config["weights"]["gan"]["fake"]
        self.adversarial_weight = config["weights"]["gan"]["adversarial"]
        self.feature_matching_weight = config["weights"]["gan"]["feature_matching"]

    def forward(self, real, fake, model):
        # Get discriminator outputs and features for both real and fake images
        real_outputs, real_features = model.discriminator_forward(real)
        fake_outputs, fake_features = model.discriminator_forward(fake)
        faked_outputs, _ = model.discriminator_forward(fake.detach())

        # Compute hinge loss for real and fake images
        real_loss = 0
        fake_loss = 0
        adversarial_loss = 0
        for fo in fake_outputs:
            adversarial_loss = adversarial_loss - torch.mean(fo)

        for real_output, fake_output in zip(real_outputs, faked_outputs):
            real_loss = real_loss + F.relu(1.0 - real_output).mean()
            fake_loss = fake_loss + F.relu(1.0 + fake_output).mean()

        adversarial_loss = adversarial_loss * self.adversarial_weight
        real_loss = real_loss * self.real_weight
        fake_loss = fake_loss * self.fake_weight    

        # Compute feature matching loss
        feature_matching_loss = 0
        # Iterate over each scale
        for scale_real_feats, scale_fake_feats in zip(real_features, fake_features):
            # For each scale, iterate over each feature map
            for real_feat, fake_feat in zip(scale_real_feats, scale_fake_feats):
                # Calculate L1 loss between corresponding features from real and fake images
                feature_matching_loss = feature_matching_loss + F.l1_loss(real_feat.detach(), fake_feat)

        feature_matching_loss = feature_matching_loss * self.feature_matching_weight

        # Normalize losses by number of scales and sum real and fake hinge losses
        total_loss = (real_loss + fake_loss + adversarial_loss) / len(real_outputs) + feature_matching_loss / sum(len(feats) for feats in real_features)
        return total_loss,{
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'feature_matching_loss': feature_matching_loss,
            'adversarial_loss': adversarial_loss
        }
    
class CycleConsistencyLoss(nn.Module):
    def __init__(self, config, emodel, scale=5.0, margin=0.2):
        super(CycleConsistencyLoss, self).__init__()
        self.weight = config["weights"]["cycle"]
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
    def __init__(self, config, face3d, arcface, emodel, gaze_model):
        super(VasaLoss, self).__init__()
        self.face3d = face3d
        self.arcface = arcface
        self.emodel = emodel
        self.gaze_model = gaze_model
        self.gaze_weight = config["weights"]["vasa"]["gaze"]
        self.arcface_weight = config["weights"]["vasa"]["arcface"]
        self.emodel_weight = config["weights"]["vasa"]["emodel"]
        self.face3d_weight = config["weights"]["vasa"]["face3d"] 

    def forward(self, giiij, gjjij, gsd, gsmod):
        batch_size = giiij.size(0)
        # Compute perceptual loss
        zi = self.emodel(giiij)
        zj = self.emodel(gjjij)


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

        # gaze_pred_1 = self.gaze_model.get_gaze(giiij)
        # gaze_pred_2 = self.gaze_model.get_gaze(gjjij)
        # assert gaze_pred_1.size(0) == batch_size
        # gaze_loss = sum(torch.norm(gaze_pred_1 - gaze_pred_2, dim=1))/batch_size * self.gaze_weight

        esd = self.arcface(gsd)
        emod = self.arcface(gsmod)

        arcloss = F.cosine_embedding_loss(esd, emod, -torch.ones(esd.size(0)).to(esd.device))
        return (arcloss + rotation_loss + cosloss, {"arcloss": arcloss, "rotationloss": rotation_loss, "cosloss": cosloss})

