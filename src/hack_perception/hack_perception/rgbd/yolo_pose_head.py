"""Frozen YOLO segmentation features with a trainable two-view RGB-D pose head."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .full_pose_nn import rotation6d


ROI_SIZE = 5


class SegmentationFeatures:
    """Capture the highest resolution feature map fed to the segmentation head."""

    def __init__(self, detector):
        network = detector.model
        self.handles = [network.register_forward_pre_hook(self._input),
                        network.model[-1].register_forward_pre_hook(self._features)]
        self.shape = self.map = None

    def _input(self, module, args):
        self.shape = tuple(args[0].shape[-2:])

    def _features(self, module, args):
        self.map = args[0][0].detach().cpu()

    def close(self):
        for handle in self.handles:
            handle.remove()

    def crops(self, raster):
        if self.map is None or self.shape is None or self.map.shape[0] != 1:
            raise RuntimeError('expected segmentation features from one RGB image')
        height, width = raster.shape
        net_h, net_w = self.shape
        gain = min(net_h / height, net_w / width)
        # Reproduce the resize and integer top/left padding used by LetterBox.
        resized_w, resized_h = round(width * gain), round(height * gain)
        left = round((net_w - resized_w) / 2 - .1)
        top = round((net_h - resized_h) / 2 - .1)
        ids, grids = [], []
        for ident in np.unique(raster):
            if not ident:
                continue
            y, x = np.nonzero(raster == ident)
            u = np.linspace(x.min() + .5, x.max() + .5, ROI_SIZE) * resized_w / width + left
            v = np.linspace(y.min() + .5, y.max() + .5, ROI_SIZE) * resized_h / height + top
            xx, yy = np.meshgrid(2 * u / net_w - 1, 2 * v / net_h - 1)
            grids.append(np.stack((xx, yy), axis=-1))
            ids.append(int(ident))
        if not ids:
            return {}
        grid = torch.as_tensor(np.stack(grids), dtype=torch.float32)
        with torch.inference_mode():
            values = F.grid_sample(self.map.float().expand(len(ids), -1, -1, -1), grid,
                                   padding_mode='border', align_corners=False).numpy()
        return dict(zip(ids, values))


def pair_features(observations, crops, ids, extra, channels):
    result = np.zeros((2, channels, ROI_SIZE, ROI_SIZE), dtype=np.float32)
    for index in range(2):
        if observations[index] is not None and extra[6 + index]:
            result[index] = crops[index][ids[index]]
    return result


class YoloPoseHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.visual = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.SiLU(), nn.Flatten(),
                                    nn.Linear(16 * ROI_SIZE**2, 64), nn.LayerNorm(64), nn.SiLU())
        self.rgbd = nn.Sequential(nn.Conv2d(8, 24, 5, 2, 2), nn.SiLU(),
                                  nn.Conv2d(24, 40, 3, 2, 1), nn.SiLU(),
                                  nn.Conv2d(40, 48, 3, 2, 1), nn.SiLU(),
                                  nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                  nn.Linear(48 * 16, 96), nn.LayerNorm(96), nn.SiLU())
        self.pose = nn.Sequential(nn.Linear(2 * (64 + 96) + 28, 192), nn.SiLU(),
                                  nn.Dropout(.1), nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 9))

    def forward(self, images, extra, yolo):
        valid = extra[:, 6:8, None]
        rgbd = self.rgbd(images.flatten(0, 1)).reshape(len(images), 2, -1)
        visual = self.visual(yolo.flatten(0, 1)).reshape(len(images), 2, -1)
        fused = torch.cat((rgbd, visual), dim=-1) * valid
        output = self.pose(torch.cat((fused.flatten(1), extra), dim=1))
        return output[:, :3], rotation6d(output[:, 3:])
