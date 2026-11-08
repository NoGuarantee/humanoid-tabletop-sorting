"""Identity head from the sensor-trained RGB-D point-template pipeline."""

import torch
from torch import nn

from .full_pose_nn import OBJECTS
from .yolo_pose_head import YoloPoseHead


class IdentityHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        base = YoloPoseHead(channels)
        self.visual, self.rgbd = base.visual, base.rgbd
        self.output = nn.Sequential(nn.Dropout(.20), nn.Linear(160, 96), nn.LayerNorm(96),
                                    nn.SiLU(), nn.Dropout(.20), nn.Linear(96, len(OBJECTS)))

    def forward(self, images, extra, neck):
        visual = self.visual(neck.flatten(0, 1))
        rgbd = self.rgbd(images.flatten(0, 1))
        logits = self.output(torch.cat((visual, rgbd), dim=1)).reshape(len(images), 2, -1)
        valid = extra[:, 6:8, None]
        return (logits*valid).sum(1)/valid.sum(1).clamp_min(1.), logits
