"""Pose refinement against surfaces reconstructed from training RGB-D samples."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def rotated_about_center(pose, axis, angle, center):
    rot = pose[:3, :3] @ Rotation.from_rotvec(np.asarray(axis)*angle).as_matrix()
    trial = pose.copy()
    trial[:3, :3] = rot
    trial[:3, 3] += (pose[:3, :3]-rot) @ center
    return trial


def radial_profile(xyz, normals):
    r = np.linalg.norm(xyz[:, :2], axis=1)
    azimuth = xyz[:, :2]/np.maximum(r[:, None], 1e-6)
    # The observed normal is projected into the axial cross section.
    n = np.column_stack(((azimuth*normals[:, :2]).sum(1), normals[:, 2]))
    norm = np.linalg.norm(n, axis=1)
    use = norm > .70
    rz = np.column_stack((r, xyz[:, 2]))
    _, ids = np.unique(np.floor(rz[use]/.00028).astype(np.int32), axis=0, return_index=True)
    return cKDTree(rz[use][ids]), n[use][ids]/norm[use][ids, None]


def trimmed(values, fraction):
    n = max(1, int(fraction*len(values)))
    return np.partition(values, n-1)[:n]


def radial_terms(profile, cloud, pose, *, full=False):
    tree, normals = profile
    rotation, origin = pose[:3, :3], pose[:3, 3]
    local = (cloud-origin) @ rotation
    radius = np.linalg.norm(local[:, :2], axis=1)
    target = np.column_stack((radius, local[:, 2]))
    distance, index = tree.query(target)
    score = np.sqrt(np.mean(trimmed(distance, .85)**2)) + .18*np.minimum(distance, .005).mean()
    if not full:
        return score
    radial = local[:, :2]/np.maximum(radius[:, None], 1e-6)
    expected = tree.data[index]
    n = normals[index]
    point = np.column_stack((radial*expected[:, :1], expected[:, 1])) @ rotation.T+origin
    normal = np.column_stack((radial*n[:, :1], n[:, 1])) @ rotation.T
    return score, distance, point, normal


def optimize_plane(pose, cloud, correspondence, pivot, *, radial=False, budget=.006):
    """Use a trimmed normal residual while limiting weak tangent modes."""
    seed = pose
    score = correspondence(pose, full=False)
    for _ in range(15):
        _, dist, points, normals = correspondence(pose, full=True)
        count = max(20, int(.88*len(dist)))
        keep = np.argpartition(dist, count-1)[:count]
        p, n = points[keep], normals[keep]
        error = np.sum(n*(cloud[keep]-p), axis=1)
        weight = np.sqrt(np.minimum(1., .001/np.maximum(np.abs(error), 1e-8)))
        jacobian = np.column_stack((n, np.cross(p-pivot, n)/.06))
        step, _, rank, _ = np.linalg.lstsq(jacobian*weight[:, None], error*weight, rcond=.035)
        if rank < (3 if radial else 4) or np.linalg.norm(step) < 2e-6:
            break
        step *= min(1., .002/max(np.linalg.norm(step[:3]), 1e-8),
                    .003/max(np.linalg.norm(step[3:]), 1e-8))
        for scale in (1., .5, .25):
            inc = Rotation.from_rotvec(step[3:]*scale/.06).as_matrix()
            trial = pose.copy()
            trial[:3, :3] = inc @ pose[:3, :3]
            trial[:3, 3] = pivot+inc @ (pose[:3, 3]-pivot)+scale*step[:3]
            if np.linalg.norm(trial[:3, 3]-seed[:3, 3]) > budget:
                continue
            value = correspondence(trial, full=False)
            if value < score-2e-7:
                pose, score = trial, value
                break
        else:
            break
    return score, pose


def refine_tube(profile, cloud, pose, center, *, budget=.020):
    pivot = pose[:3, :3] @ center + pose[:3, 3]
    return optimize_plane(pose, cloud, lambda trial, full: radial_terms(profile, cloud, trial, full=full),
                          pivot, radial=True, budget=budget)


def learned_scores(net, poses, image, extra, anchor, name, *, slot=None):
    import torch
    from .full_pose_nn import OBJECTS, SCALE_M

    inputs = extra.copy()
    inputs[8:8+len(OBJECTS)] = 0
    inputs[8+OBJECTS.index(name)] = 1
    if slot is not None:
        inputs[6+(1-slot)] = 0
    offset = torch.from_numpy(np.asarray((poses[None, :, :3, 3]-anchor)/SCALE_M, dtype=np.float32))
    rotation = torch.from_numpy(np.asarray(poses[None, :, :3, :3], dtype=np.float32))
    with torch.inference_mode():
        common = (torch.from_numpy(image[None]), torch.from_numpy(inputs[None]))
        if name.startswith("falcon"):
            score = net(*common, rotation[..., 2], offset, torch.tensor([int(name == "falcon15")]))
        else:
            score = net(*common, rotation, offset)
    return score[0].numpy()


def stable_tube(net, profile, bounds, incumbent, clouds, image, extra, anchor, name, clipped):
    """Resolve the end assignment near an already fitted physical tube center."""
    low, high = bounds
    center = np.array([0., 0., (low[2]+high[2])/2])
    slots = [i for i, cloud in enumerate(clouds) if cloud is not None and len(cloud) >= 30]
    if not slots:
        return [incumbent, incumbent], {}
    cloud = np.concatenate([clouds[i] for i in slots])
    opposite = rotated_about_center(incumbent, [1., 0., 0.], np.pi, center)
    before = radial_terms(profile, cloud, incumbent)
    poses, costs = [], []
    pivot = incumbent[:3, :3] @ center+incumbent[:3, 3]
    for seed in (incumbent, opposite):
        value, proposal = refine_tube(profile, cloud, seed, center, budget=.004)
        q = proposal[:3, :3] @ center+proposal[:3, 3]
        if np.linalg.norm(q-pivot) > .0045 or seed[:3, 2] @ proposal[:3, 2] < .995:
            value, proposal = radial_terms(profile, cloud, seed), seed
        poses.append(proposal)
        costs.append(value)
    poses = np.asarray(poses)
    scores = learned_scores(net, poses, image, extra, anchor, name, slot=None if len(slots) == 2 else slots[0])
    margin = float(scores[1]-scores[0])
    signed = [learned_scores(net, poses, image, extra, anchor, name, slot=i) for i in slots]
    per_view = [float(v[1]-v[0]) for v in signed]
    projection = (cloud-incumbent[:3, 3]) @ incumbent[:3, 2]
    span = float(np.ptp(np.quantile(projection, (.02, .98))))/float(high[2]-low[2])
    support = span >= .60 and max(per_view) >= 2.4 and min(per_view) >= -1.0
    pose = incumbent
    if (support and margin >= 2.0 and costs[1] <= min(costs[0]+.00020, .0011)
            and all(radial_terms(profile, clouds[i], poses[1])
                    <= radial_terms(profile, clouds[i], poses[0])+.00027 for i in slots)):
        pose = poses[1]
    elif before-costs[0] > .00012 and np.linalg.norm(poses[0, :3, 3]-incumbent[:3, 3]) <= .0025:
        pose = poses[0]
    needs_reacquire = (span < .30 or
                       (len(slots) == 1 and clipped[slots[0]] and span < .60
                        and abs(margin) < 2.4))
    return [pose, pose], dict(stable_end_scores=per_view, stable_end_margin=margin,
                              stable_end_depth_mm=[float(1000*x) for x in costs],
                              stable_end_changed=bool(pose[:3, 2] @ incumbent[:3, 2] < 0),
                              stable_end_span_fraction=span, tube_needs_reacquire=needs_reacquire)
