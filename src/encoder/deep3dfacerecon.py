"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import os
import torch
from collections import OrderedDict
from torch import nn
import cv2

import trimesh
from abc import ABC, abstractmethod

from typing import Type, Any, Callable, Union, List, Optional

import torch.nn.functional as F
from scipy.io import loadmat, savemat
from argparse import ArgumentTypeError 

def perspective_projection(focal, center):
    # return p.T (N, 3) @ (3, 3) 
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()

class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)


def draw_landmarks(img, landmark, save_name):
    landmark = landmark
    lm_img = np.zeros([img.shape[0], img.shape[1], 3])
    lm_img[:] = img.astype(np.float32)
    landmark = np.round(landmark).astype(np.int32)

    for i in range(len(landmark)):
        for j in range(-1, 1):
            for k in range(-1, 1):
                if img.shape[0] - 1 - landmark[i, 1]+j > 0 and \
                        img.shape[0] - 1 - landmark[i, 1]+j < img.shape[0] and \
                        landmark[i, 0]+k > 0 and \
                        landmark[i, 0]+k < img.shape[1]:
                    lm_img[img.shape[0] - 1 - landmark[i, 1]+j, landmark[i, 0]+k,
                           :] = np.array([0, 0, 255])
    lm_img = lm_img.astype(np.uint8)

    cv2.imwrite(save_name, lm_img)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')



class MeshRenderer(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224,
                use_opengl=True):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(
                torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.use_opengl = use_opengl
        self.ctx = None
    
    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 


        vertex_ndc = vertex @ ndc_proj.t()
        if self.ctx is None:
            if self.use_opengl:
                self.ctx = dr.RasterizeGLContext(device=device)
                ctx_str = "opengl"
            else:
                self.ctx = dr.RasterizeCudaContext(device=device)
                ctx_str = "cuda"
            print("create %s ctx on device cuda:%d"%(ctx_str, device.index))
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.ctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        depth, _ = dr.interpolate(vertex.reshape([-1,4])[...,2].unsqueeze(1).contiguous(), rast_out, tri) 
        depth = depth.permute(0, 3, 1, 2)
        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image
        
        return mask, depth, image


class ParametricFaceModel:
    def __init__(self, 
                bfm_folder='./BFM', 
                recenter=True,
                camera_distance=10.,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                focal=1015.,
                center=112.,
                is_train=True,
                default_name='BFM_model_front.mat'):
        
        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            print("what's up with this???")
            util.transferBFM09(bfm_folder)
        model = loadmat(os.path.join(bfm_folder, default_name))
        # mean face shape. [3*N,1]
        self.mean_shape = model['meanshape'].astype(np.float32)
        # identity basis. [3*N,80]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = model['exBase'].astype(np.float32)
        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = model['meantex'].astype(np.float32)
        # texture basis. [3*N,80]
        self.tex_base = model['texBase'].astype(np.float32)
        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1
        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1
        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        if is_train:
            # vertex indices for small face region to compute photometric error. starts from 0.
            self.front_mask = np.squeeze(model['frontmask2_idx']).astype(np.int64) - 1
            # vertex indices for each face from small face region. starts from 0. [f,3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # vertex indices for pre-defined skin region to compute reflectance loss
            self.skin_mask = np.squeeze(model['skinmask'])
        
        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        self.persc_proj = perspective_projection(focal, center)
        self.device = 'cpu'
        self.camera_distance = camera_distance
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)
        

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    
    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])
    

    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.tex_base, tex_coeff) + self.mean_tex
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.reshape([batch_size, -1, 3])


    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm


    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    
    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)


    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj


    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)


    def get_landmarks(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)

        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """  
        return face_proj[:, self.keypoints]

    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }
    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation(coef_dict['angle'])


        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)
        
        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark


class BaseModel(ABC):
    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device('cpu') 
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.parallel_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            print("=="*10)
            print("ERROR SHOULD NOT BE TRAINING")
            print("=="*10)
            #  self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)
 
            
        # self.print_networks(opt.verbose)

    def parallelize(self, convert_sync_batchnorm=True):
        if not self.opt.use_ddp:
            for name in self.parallel_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))
        else:
            for name in self.model_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    if convert_sync_batchnorm:
                        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
                    setattr(self, name, torch.nn.parallel.DistributedDataParallel(module.to(self.device),
                        device_ids=[self.device.index], 
                        find_unused_parameters=True, broadcast_buffers=True))
            
            # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
            for name in self.parallel_names:
                if isinstance(name, str) and name not in self.model_names:
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))
            
        # put state_dict of optimizer to gpu device
        if self.opt.phase != 'test':
            if self.opt.continue_train:
                for optim in self.optimizers:
                    for state in optim.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)

    def data_dependent_initialize(self, data):
        pass

    def train(self):
        """Make models train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        """Make models eval mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self, name='A'):
        """ Return image paths that are used to load current data"""
        return self.image_paths if name =='A' else self.image_paths_B

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)[:, :3, ...]
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        save_filename = 'epoch_%s.pth' % (epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        
        save_dict = {}
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel) or isinstance(net,
                        torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                save_dict[name] = net.state_dict()
                

        for i, optim in enumerate(self.optimizers):
            save_dict['opt_%02d'%i] = optim.state_dict()

        for i, sched in enumerate(self.schedulers):
            save_dict['sched_%02d'%i] = sched.state_dict()
        
        torch.save(save_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if self.opt.isTrain and self.opt.pretrained_name is not None:
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
        else:
            load_dir = self.save_dir    
        load_filename = 'epoch_%s.pth' % (epoch)
        load_path = os.path.join(load_dir, load_filename)
        state_dict = torch.load(load_path, map_location=self.device)
        print('loading the model from %s' % load_path)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])
        
        if self.opt.phase != 'test':
            if self.opt.continue_train:
                print('loading the optim from %s' % load_path)
                for i, optim in enumerate(self.optimizers):
                    optim.load_state_dict(state_dict['opt_%02d'%i])

                try:
                    print('loading the sched from %s' % load_path)
                    for i, sched in enumerate(self.schedulers):
                        sched.load_state_dict(state_dict['sched_%02d'%i])
                except:
                    print('Failed to load schedulers, set schedulers according to epoch count manually')
                    for i, sched in enumerate(self.schedulers):
                        sched.last_epoch = self.opt.epoch_count - 1
                    

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        use_last_fc: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.use_last_fc = use_last_fc
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.use_last_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.use_last_fc:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)



def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


func_dict = {
    'resnet50': (resnet50, 2048)
}

def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

class ReconNetWrapper(nn.Module):
    fc_dim=257
    def __init__(self, net_recon, use_last_fc=False, init_path=None):
        super(ReconNetWrapper, self).__init__()
        self.use_last_fc = use_last_fc
        if net_recon not in func_dict:
            return  NotImplementedError('network [%s] is not implemented', net_recon)
        func, last_dim = func_dict[net_recon]
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)
        if init_path and os.path.isfile(init_path):
            state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
            backbone.load_state_dict(state_dict)
            print("loading init net_recon %s from %s" %(net_recon, init_path))
        self.backbone = backbone
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                conv1x1(last_dim, 80, bias=True), # id layer
                conv1x1(last_dim, 64, bias=True), # exp layer
                conv1x1(last_dim, 80, bias=True), # tex layer
                conv1x1(last_dim, 3, bias=True),  # angle layer
                conv1x1(last_dim, 27, bias=True), # gamma layer
                conv1x1(last_dim, 2, bias=True),  # tx, ty
                conv1x1(last_dim, 1, bias=True)   # tz
            ])
            for m in self.final_layers:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        return x

def define_net_recon(net_recon, use_last_fc=False, init_path=None):
    return ReconNetWrapper(net_recon, use_last_fc=use_last_fc, init_path=init_path)



class FaceReconModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='BFM')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)
        parser.add_argument('--use_opengl', type=str2bool, nargs='?', const=True, default=True, help='use opengl context or not')

        # if is_train:
        #     # training parameters
        #     parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'], help='face recog network structure')
        #     parser.add_argument('--net_recog_path', type=str, default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
        #     parser.add_argument('--use_crop_face', type=util.str2bool, nargs='?', const=True, default=False, help='use crop mask for photo loss')
        #     parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False, help='use predefined M for predicted face')

        #     
        #     # augmentation parameters
        #     parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')
        #     parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')
        #     parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')

        #     # loss weights
        #     parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
        #     parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
        #     parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
        #     parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
        #     parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
        #     parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
        #     parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
        #     parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
        #     parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')



        # opt, _ = parser.parse_known_args()
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        # if is_train:
        #     parser.set_defaults(
        #         use_crop_face=True, use_predef_M=False
        #     )
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.net_recon = define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        )

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )
        
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        # self.renderer = MeshRenderer(
        #     rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center), use_opengl=opt.use_opengl
        # )
        self.renderer = None

        if self.isTrain:
            print("=="*10)
            print("ERROR SHOULD NOT BE TRAINING")
            print("=="*10)
            
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def forward(self):
        output_coeff = self.net_recon(self.input_img)
        self.facemodel.to(self.device)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)
        # self.pred_mask, _, self.pred_face = self.renderer(
        #     self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
        
        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)


    def compute_losses(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        assert self.net_recog.training == False
        trans_m = self.trans_m
        if not self.opt.use_predef_M:
            # trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])
            print("FAILFAILFAIL")

        pred_feat = self.net_recog(self.pred_face, trans_m)
        gt_feat = self.net_recog(self.input_img, self.trans_m)
        self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)

        face_mask = self.pred_mask
        if self.opt.use_crop_face:
            face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)
        
        face_mask = face_mask.detach()
        self.loss_color = self.opt.w_color * self.comupte_color_loss(
            self.pred_face, self.input_img, self.atten_mask * face_mask)
        
        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        self.loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(self.pred_tex, self.facemodel.skin_mask)

        self.loss_all = self.loss_feat + self.loss_color + self.loss_reg + self.loss_gamma \
                        + self.loss_lm + self.loss_reflc
            

    def optimize_parameters(self, isTrain=True):
        self.forward()               
        self.compute_losses()
        """Update network weights; it will be called in every training iteration."""
        if isTrain:
            self.optimizer.zero_grad()  
            self.loss_all.backward()         
            self.optimizer.step()        

    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
            
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)

    def save_mesh(self, name):

        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8), process=False)
        mesh.export(name)

    def save_coeff(self,name):

        pred_coeffs = {key:self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:,:,0],self.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        savemat(name,pred_coeffs)


def get_face_recon_model(opt):
    model = FaceReconModel(opt)
    model.setup(opt)
    return model