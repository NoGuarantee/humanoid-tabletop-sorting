"""Score the two Falcon end assignments from observed RGB-D in an axial frame."""

import torch
from torch import nn
from torch.nn import functional as F


class TubeEndRanker(nn.Module):
    def __init__(self, lengths, radii):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0., 1., 9))
        self.register_buffer("lengths", torch.tensor(lengths, dtype=torch.float32))
        self.register_buffer("radii", torch.tensor(radii, dtype=torch.float32))
        self.point = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(9*25+3, 88), nn.LayerNorm(88), nn.SiLU(),
                                    nn.Dropout(.18), nn.Linear(88, 1))

    def forward(self, images, extra, axes, offsets, categories):
        image = images[..., 1::3, 1::3].flatten(-2).transpose(-2, -1)
        xyz = image[:, None, :, :, 3:6]-offsets[:, :, None, None]
        longitudinal = torch.einsum("bhvpc,bhc->bhvp", xyz, axes)
        radial = (xyz.square().sum(-1)-longitudinal.square()).clamp_min(0.).sqrt()
        axial = (longitudinal/self.lengths[categories, None, None, None]).clamp(-.30, 1.30)
        radial = (radial/self.radii[categories, None, None, None]).clamp(0., 2.)
        rgb = image[:, None, :, :, :3].expand(-1, axes.shape[1], -1, -1, -1)
        encoded = self.point(torch.cat((rgb, axial[..., None], radial[..., None], rgb*axial[..., None]), dim=-1))
        valid = image[:, None, :, :, 7] > .5
        weights = (1.-8.*(axial[..., None]-self.centers).abs()).clamp_min(0.)*valid[..., None]
        count = weights.sum(3)
        summary = torch.einsum("bhvpf,bhvpk->bhvkf", encoded, weights)/count[..., None].clamp_min(1.)
        coverage = (count/30.).clamp(0., 1.)
        camera_dir = torch.einsum("bvc,bhc->bhv", extra[:, :6].reshape(len(images), 2, 3), axes)
        kinds = F.one_hot(categories, 2)[:, None, None].expand(-1, axes.shape[1], 2, -1)
        value = self.output(torch.cat((summary.flatten(-2), coverage, camera_dir[..., None], kinds), dim=-1)).squeeze(-1)
        active = extra[:, None, 6:8] * valid.any(3)
        return (value*active).sum(2)/active.sum(2).clamp_min(1.)
