"""Recognize box variants from two observed RGB-D crops and frozen YOLO features."""

import torch
from torch import nn

from .yolo_pose_head import YoloPoseHead

PACKS = (
    ('Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g',
     'Brother_Ink_Cartridge_Magenta_LC75M', 'Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack'),
    ('Nestle_Nips_Hard_Candy_Peanut_Butter', 'Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g', '009_gelatin_box'),
)

NAMES = PACKS[0] + PACKS[1]


class PairedBoxIdentity(nn.Module):
    def __init__(self, channels):
        super().__init__()
        base = YoloPoseHead(channels)
        self.visual, self.rgbd = base.visual, base.rgbd
        self.classifier = nn.Sequential(nn.Dropout(.25), nn.Linear(160, 96), nn.LayerNorm(96),
                                        nn.SiLU(), nn.Dropout(.20), nn.Linear(96, len(NAMES)))

    def forward(self, images, extra, neck):
        rgbd = self.rgbd(images.flatten(0, 1)).reshape(len(images), 2, -1)
        visual = self.visual(neck.flatten(0, 1)).reshape(len(images), 2, -1)
        per_view = self.classifier(torch.cat((rgbd, visual), dim=-1))
        active = extra[:, 6:8, None]
        fused = (per_view*active).sum(1)/active.sum(1).clamp_min(1.)
        return fused, per_view
