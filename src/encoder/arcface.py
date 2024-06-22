import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from torch import nn
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from torchvision import transforms as trans
from collections import namedtuple

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Backbone(nn.Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      nn.BatchNorm2d(64), 
                                      nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), 
                                       nn.Dropout(drop_ratio),
                                       nn.Flatten(),
                                       nn.Linear(512 * 7 * 7, 512),
                                       nn.BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)
        self.transforms = trans.Compose([
                    trans.Resize((112, 112)),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    
    def forward(self,x):
        x = self.transforms(x)
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x

def get_config(training = True):
    conf = {}
    conf["data_path"] = './data'
    conf["work_path"] = './work_space/'
    conf["model_path"] = conf["work_path"]+'/models'
    conf["log_path"] = conf["work_path"]+ '/log'
    conf["save_path"] = conf["work_path"]+ '/save'
    conf["input_size"] = [112,112]
    conf["embedding_size"] = 512
    conf["use_mobilfacenet"] = False
    conf["net_depth"] = 50
    conf["drop_ratio"] = 0.6
    conf["net_mode"] = 'ir_se' # or 'ir'
    conf["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["test_transform"] = trans.Compose([
                    trans.Resize((112, 112)),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf["data_mode"] = 'emore'
    conf["vgg_folder"] = conf["data_path"]+'/faces_vgg_112x112'
    conf["ms1m_folder"] = conf["data_path"]+'/faces_ms1m_112x112'
    conf["emore_folder"] = conf["data_path"]+'/faces_emore'
    conf["batch_size"] = 100 # irse net depth 50 

    conf["facebank_path"] = conf["data_path"]+'/facebank'
    conf["threshold"] = 1.5
    conf["face_limit"] = 10 
    #when inference, at maximum detect 10 faces in one image, my laptop is slow
    conf["min_face_size"] = 30 
    # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf

def get_time(self):
        return "womp womp"


class face_learner(object):

    def __init__(self):
        conf = get_config(training=False)
        print(conf)

        self.model = Backbone(conf["net_depth"], conf["drop_ratio"], conf["net_mode"]).to(conf["device"])
        print('{}_{} model generated'.format(conf["net_mode"], conf["net_depth"]))
        
        self.threshold = 0.01
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf["save_path"]
        else:
            save_path = conf["model_path"]
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
    
    def load_state(self, fixed_str, from_save_folder=False, model_only=False):
        conf = get_config(training=False)
        if from_save_folder:
            save_path = conf["save_path"]
        else:
            save_path = conf["model_path"]            
        self.model.load_state_dict(torch.load(fixed_str))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path+'/head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path+'/optimizer_{}'.format(fixed_str)))

def get_model_arcface(model_path):
    fl = face_learner()
    fl.load_state(model_path, True, True)
    fl.model.eval()
    for p in fl.model.parameters():
        p.requires_grad = False
    return fl.model

        
if __name__ == "__main__":
    from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
    import matplotlib.pyplot as plt
    import torch
    import numpy as np


    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    model_path = "/home/jalnyn/git/talkinghead/models/arcface2/model_ir_se50.pth"
    print(video_dataset[0].shape)
    print(video_dataset[0][0].shape)
    learner = get_model_arcface(model_path)
    frame0 = video_dataset[0][0].permute(2, 0, 1)
    frame1 = video_dataset[0][1].permute(2, 0, 1)
    frame2 = video_dataset[1][0].permute(2, 0, 1)
    conf = get_config(training=False)
    print(frame0.shape, frame1.shape)
    emb1 = learner((frame0).unsqueeze(0).to("cuda"))
    emb2 = learner((frame1).unsqueeze(0).to("cuda"))
    emb3 = learner((frame2).unsqueeze(0).to("cuda"))
    print(emb1)
    print(emb2)
    print(torch.norm(emb1 - emb2))
    print(torch.norm(emb1 - emb3))
