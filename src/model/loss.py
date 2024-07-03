import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from facenet_pytorch import InceptionResnetV1
from src.model.discriminator import MultiScalePatchDiscriminator
from torchvision.transforms import Normalize

import lpips

class SimpleLoss(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLoss, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class PerceptualLoss(nn.Module):
    def __init__(self, config):
        super(PerceptualLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.lpips = lpips.LPIPS(net='vgg').to(self.device)
        self.vggface = InceptionResnetV1(pretrained='vggface2').eval()
        
        for param in self.vggface.parameters():
            param.requires_grad = False

        self.lpips_weight = config["weights"]["perceptual"]["lpips"]


    def forward(self, driver, pred):
        driver = F.interpolate(driver, size=(224, 224), mode='bilinear')
        pred = F.interpolate(pred, size=(224, 224), mode='bilinear')

        lpips_loss = self.lpips(pred, driver).mean() * self.lpips_weight

        # Return individual losses along with the total
        total_loss = lpips_loss
        return total_loss, {
            'lpips': lpips_loss,
        }


class GANLoss(nn.Module):
    def __init__(self, config, model):
        super(GANLoss, self).__init__()
        self.model = model 
        self.real_weight = config["weights"]["gan"]["real"] 
        self.fake_weight = config["weights"]["gan"]["fake"]
        self.adversarial_weight = config["weights"]["gan"]["adversarial"]
        self.feature_matching_weight = config["weights"]["gan"]["feature_matching"]

    def forward(self, real, fake, alpha, steps):
        if real.size() != fake.size():
            # Downsample driver to the size of pred
            # Assuming pred and driver are 4D tensors [batch, channels, height, width]
            real = F.interpolate(real, size=(fake.size(2), fake.size(3)), mode='bilinear')

        # Get discriminator outputs and features for both real and fake images
        real_outputs = self.model.discriminator_forward(real, alpha, steps)
        fake_outputs = self.model.discriminator_forward(fake, alpha, steps)
        faked_outputs = self.model.discriminator_forward(fake.detach(), alpha, steps)

        # Compute hinge loss for real and fake images
        real_loss = 0
        fake_loss = 0
        adversarial_loss = 0
        adversarial_loss = adversarial_loss - torch.mean(fake_outputs)

        real_loss = real_loss + F.relu(1.0 - real_outputs).mean()
        fake_loss = fake_loss + F.relu(1.0 + faked_outputs).mean()

        adversarial_loss = adversarial_loss * self.adversarial_weight
        real_loss = real_loss * self.real_weight
        fake_loss = fake_loss * self.fake_weight    

        # Normalize losses by number of scales and sum real and fake hinge losses
        total_loss = (real_loss + fake_loss + adversarial_loss) / len(real_outputs)
        return total_loss,{
            'real_loss': real_loss,
            'fake_loss': fake_loss,
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

if __name__ == '__main__':
    #### TESTING LOSSES
    from loss import PerceptualLoss, GANLoss
    from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    device = "cuda"

    input_data = video_dataset[0][0:2].to(device)
    input_data2 = video_dataset[0][2:4].to(device)
    input_data_backup = input_data.clone()  # Backup to check for modifications
    input_data_clone = input_data.clone()  # Clone to prevent modification
    input_data_clone.requires_grad = False

    import yaml
    import torch.optim as optim

    with open('config/local_train.yaml', 'r') as file:
        config = yaml.safe_load(file)

    from src.model.portrait import Portrait

    p = Portrait(config=config)

    perceptionloss = PerceptualLoss(config)
    ganloss = GANLoss(config, p)

    ### Make sure losses are the same before and after backwards passes
    def test_loss(loss_fn, t_data1, t_data2, model):
        # Clone the input data and set requires_grad
        input_data1 = t_data1.clone().detach().requires_grad_(True).to(model.device)
        input_data2 = t_data2.clone().detach().requires_grad_(True).to(model.device)

        # Zero the gradients
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()

        # Compute the initial loss
        initial_loss = loss_fn(input_data1, input_data2)[0]
        
        # Perform the backward pass
        initial_loss.backward(retain_graph=True)


        # Check if the gradients are consistent
        grad_input_data1 = input_data1.grad.clone()
        grad_input_data2 = input_data2.grad.clone()
        
        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

        # Compute the loss again
        input_data1 = t_data1.clone().detach().requires_grad_(True).to(model.device)
        input_data2 = t_data2.clone().detach().requires_grad_(True).to(model.device)

        new_loss = loss_fn(input_data1, input_data2)[0]
        new_loss.backward(retain_graph=True)

        # Assert that the loss did not change
        assert torch.allclose(initial_loss, new_loss), "Loss changed after optimizer step"

        # Check if the gradients are zero after optimizer step
        assert torch.allclose(input_data1.grad, grad_input_data1), "Gradients changed for input_data1"
        assert torch.allclose(input_data2.grad, grad_input_data2), "Gradients changed for input_data2"

    test_loss(perceptionloss, input_data, input_data2, p)

    print("All checks passed successfully. loss outputs are consistent after training and loading.")
