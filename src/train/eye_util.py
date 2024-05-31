

import torch
import torch.nn as nn
import dlib

from torch import nn
import numpy as np
import cv2

Pool = nn.MaxPool2d


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


def softargmax2d(input, beta=100, dtype=torch.float32):
    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)))
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)))

    device = input.get_device()
    if device >= 0:
        indices_r = indices_r.to(device)
        indices_c = indices_c.to(device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result.type(dtype)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = nn.functional.interpolate(low3, x.shape[2:], mode='bilinear')
        return up1 + up2

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class EyeSample:
    def __init__(self, orig_img, img, is_left, transform_inv, estimated_radius):
        self._is_left = is_left
        self._transform_inv = transform_inv
        self._estimated_radius = estimated_radius
    @property
    def orig_img(self):
        return self._orig_img

    @property
    def img(self):
        return self._img

    @property
    def is_left(self):
        return self._is_left

    @property
    def transform_inv(self):
        return self._transform_inv

    @property
    def estimated_radius(self):
        return self._estimated_radius


class EyePrediction():
    def __init__(self, eye_sample: EyeSample, landmarks, gaze):
        self._eye_sample = eye_sample
        self._landmarks = landmarks
        self._gaze = gaze

    @property
    def eye_sample(self):
        return self._eye_sample

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def gaze(self):
        return self._gaze


class EyeNet(nn.Module):
    def __init__(self, nstack, nfeatures, nlandmarks, bn=False, increase=0, **kwargs):
        super(EyeNet, self).__init__()

        self.img_w = 160
        self.img_h = 96
        self.nstack = nstack
        self.nfeatures = nfeatures
        self.nlandmarks = nlandmarks

        self.heatmap_w = self.img_w / 2
        self.heatmap_h = self.img_h / 2

        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 1, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, nfeatures)
        )

        self.pre2 = nn.Sequential(
            Conv(nfeatures, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, nfeatures)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, nfeatures, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(nfeatures, nfeatures),
                Conv(nfeatures, nfeatures, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([Conv(nfeatures, nlandmarks, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(nfeatures, nfeatures) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(nlandmarks, nfeatures) for i in range(nstack - 1)])

        self.gaze_fc1 = nn.Linear(in_features=int(nfeatures * self.img_w * self.img_h / 64 + nlandmarks*2), out_features=256)
        self.gaze_fc2 = nn.Linear(in_features=256, out_features=2)

        self.nstack = nstack
        self.heatmapLoss =None 
        self.landmarks_loss = nn.MSELoss()
        self.gaze_loss = nn.MSELoss()

    def forward(self, imgs):
        # imgs of size 1,ih,iw
        x = imgs.unsqueeze(1)
        x = self.pre(x)

        gaze_x = self.pre2(x)
        gaze_x = gaze_x.flatten(start_dim=1)

        combined_hm_preds = []
        for i in torch.arange(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        heatmaps_out = torch.stack(combined_hm_preds, 1)

        # preds = N x nlandmarks * heatmap_w * heatmap_h
        landmarks_out = softargmax2d(preds)  # N x nlandmarks x 2

        # Gaze
        gaze = torch.cat((gaze_x, landmarks_out.flatten(start_dim=1)), dim=1)
        gaze = self.gaze_fc1(gaze)
        gaze = nn.functional.relu(gaze)
        gaze = self.gaze_fc2(gaze)

        return heatmaps_out, landmarks_out, gaze

    def calc_loss(self, combined_hm_preds, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[:, i, :], heatmaps))

        heatmap_loss = torch.stack(combined_loss, dim=1)
        landmarks_loss = self.landmarks_loss(landmarks_pred, landmarks)
        gaze_loss = self.gaze_loss(gaze_pred, gaze)

        return torch.sum(heatmap_loss), landmarks_loss, 1000 * gaze_loss


class GazeDetector:
    def __init__(self, landmarks_detector_path, eyenet_model_path, device='cpu'):
        self.landmarks_detector = dlib.shape_predictor(landmarks_detector_path)
        self.eyenet = torch.load(eyenet_model_path, map_location=device)
        self.device = device

    def shape_to_np(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords


    def detect_landmarks(self, frame):
        rectangle = dlib.rectangle(0, 0, frame.shape[1], frame.shape[0])
        face_landmarks = self.landmarks_detector(frame, rectangle)
        return self.shape_to_np(face_landmarks)

    def segment_eyes(self, frame, landmarks, ow=160, oh=96):
        eyes = []
        for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
            x1, y1 = landmarks[corner1, :]
            x2, y2 = landmarks[corner2, :]
            eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
            if eye_width == 0.0:
                return eyes

            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            scale = ow / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            estimated_radius = 0.5 * eye_width * scale
            center_mat = np.asmatrix(np.eye(3))
            center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
            transform_mat = center_mat * scale_mat * translate_mat

            eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
            eye_image = cv2.equalizeHist(eye_image)

            if is_left:
                eye_image = np.fliplr(eye_image)
            eyes.append(EyeSample(orig_img=frame.copy(), img=eye_image, transform_inv=transform_mat.I, is_left=is_left, estimated_radius=estimated_radius))
        return eyes

    def run_eyenet(self, eyes , ow=160, oh=96):
        result = []
        for eye in eyes:
            with torch.no_grad():
                x = torch.tensor([eye.img], dtype=torch.float32).to(self.device)
                _, landmarks, gaze = self.eyenet.forward(x)
                landmarks = np.asarray(landmarks.cpu().numpy()[0])
                gaze = np.asarray(gaze.cpu().numpy()[0])
                landmarks = landmarks * np.array([oh/48, ow/80])
                temp = np.zeros((34, 3))
                if eye.is_left:
                    temp[:, 0] = ow - landmarks[:, 1]
                else:
                    temp[:, 0] = landmarks[:, 1]
                temp[:, 1] = landmarks[:, 0]
                temp[:, 2] = 1.0
                landmarks = temp
                landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
                result.append(EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze))
        return result

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return self.forward(*args, **kwds)

    def forward(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self.detect_landmarks(gray)
        
        if landmarks is None:
            return None, None
        
        eye_samples = self.segment_eyes(gray, landmarks)
        eye_preds = self.run_eyenet(eye_samples)
        left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
        right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

        left_eye = left_eyes[0] if left_eyes else None
        right_eye = right_eyes[0] if right_eyes else None

        return left_eye, right_eye

def get_eye_detector(landmarks_detector_path, eyenet_model_path, device='cuda'):
    return GazeDetector(landmarks_detector_path, eyenet_model_path, device)

if __name__ == "__main__":
    from src.dataloader import VideoDataset, transform  # Import the dataset class and transformation
    import matplotlib.pyplot as plt
    import torch
    import numpy as np


    video_dataset = VideoDataset(root_dir='./dataset/mp4', transform=transform)
    model_path = "/home/jalnyn/git/talkinghead/models/gaze_est/checkpoint.pt"
    shape_path = "/home/jalnyn/git/talkinghead/models/gaze_est/shape_predictor_5_face_landmarks.dat"
    print(video_dataset[0].shape)
    print(video_dataset[0][0].shape)
    model = get_eye_detector(shape_path, model_path)
    exit(1)
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

