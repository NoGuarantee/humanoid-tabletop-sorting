"""Estimate pelvis pose using RGB-D-trained point templates and frozen YOLO features."""

import hashlib
import json
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import depth_mask_to_pelvis
from .seeded_icp import refine


def setup(folder, catalog):
    index = json.loads((folder/"catalog.json").read_text())
    for name, meta in index["objects"].items():
        path = folder/f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"point template checksum mismatch for {name}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"xyz", "normals", "votes"}:
                raise ValueError("expected an RGB-D point template")
            xyz, normals = archive["xyz"], archive["normals"]
        low, high = np.quantile(xyz, (.005, .995), axis=0)
        _, ids = np.unique(np.floor(xyz/.004).astype(np.int32), axis=0, return_index=True)
        coarse = xyz[ids]
        catalog[name] = (xyz, normals, cKDTree(xyz), coarse, cKDTree(coarse), low, high)


def measure(depth, mask, k, camera, calibrated):
    interior = binary_erosion(mask)
    if np.count_nonzero(interior & (depth >= 150) & (depth <= 6000)) < 45:
        interior = mask
    points = depth_mask_to_pelvis(depth, interior, k, camera)
    if calibrated and len(points):
        optical = (points-camera[:3, 3]) @ camera[:3, :3]
        z = optical[:, 2]
        correction = optical/z[:, None]*.0005
        correction[:, :2] += .5*(z[:, None]+.0005)/np.diag(k)[:2]
        points += correction @ camera[:3, :3].T
    chosen = np.linspace(0, len(points)-1, min(650, len(points)), dtype=int)
    return points[chosen]


def pose_error(tree, cloud, pose, fraction=.84):
    distances = tree.query((cloud-pose[:3, 3]) @ pose[:3, :3])[0]
    count = int(fraction*len(distances))
    keep = np.partition(distances, count)[:count]
    return np.sqrt(np.mean(keep**2))


def upright_candidates(cloud, network, low, high, plane):
    normal, offset = plane
    align = Rotation.align_vectors([normal], [[0., 0., 1.]])[0].as_matrix()
    middle = (low+high)*.5
    for angle in range(0, 360, 15):
        rot = align @ Rotation.from_euler("z", angle, degrees=True).as_matrix()
        observed = cloud @ rot
        ol, oh = np.quantile(observed[:, :2], (.025, .975), axis=0)
        shifts = np.stack((ol-low[:2], oh-high[:2], .5*(ol+oh)-middle[:2]))
        for a, b in ((2, 2), (0, 0), (0, 1), (1, 0), (1, 1)):
            t = rot @ [shifts[a, 0], shifts[b, 1], 0.]
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rot, t + (-offset-low[2]-normal @ t)*normal
            yield pose
    if network[:3, 2] @ normal > .6:
        seed = network.copy()
        seed[:3, 3] -= (normal @ seed[:3, 3]+offset+low[2])*normal
        yield seed


def axial_candidates(cloud, seed, low, high):
    centered = cloud-np.mean(cloud, axis=0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    midpoint = (low+high)*.5
    for direction in (axis, -axis, seed[:3, 2], -seed[:3, 2]):
        rot = Rotation.align_vectors([direction], [[0., 0., 1.]])[0].as_matrix()
        qlo, qhi = np.quantile(cloud @ rot, (.03, .97), axis=0)
        t = (qlo+qhi)/2-midpoint
        for end in (t[2], qlo[2]-low[2], qhi[2]-high[2]):
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rot, rot @ [t[0], t[1], end]
            yield pose


def point_plane(model, normals, tree, cloud, initial):
    pose = initial.copy()
    value = pose_error(tree, cloud, pose)
    for _ in range(14):
        r, t = pose[:3, :3], pose[:3, 3]
        distance, index = tree.query((cloud-t) @ r)
        keep = np.argsort(distance)[:int(.86*len(distance))]
        q, n = model[index[keep]] @ r.T, normals[index[keep]] @ r.T
        center = np.median(q, axis=0)
        pivot = center+t
        gap = np.sum(n*(cloud[keep]-q-t), axis=1)
        weight = np.sqrt(np.minimum(1., .0013/np.maximum(np.abs(gap), 1e-9)))
        system = np.column_stack((n, np.cross(q-center, n)/.06))
        step, _, rank, _ = np.linalg.lstsq(system*weight[:, None], gap*weight, rcond=.012)
        if rank < 4 or np.linalg.norm(step) < 1e-6:
            break
        step *= min(1., .005/max(np.linalg.norm(step[:3]), 1e-9), .006/max(np.linalg.norm(step[3:]), 1e-9))
        for size in (1., .5, .25):
            inc = Rotation.from_rotvec(size*step[3:]/.06).as_matrix()
            candidate = pose.copy()
            candidate[:3, :3] = inc @ r
            candidate[:3, 3] = pivot+inc @ (t-pivot)+size*step[:3]
            trial = pose_error(tree, cloud, candidate)
            if trial < value-1e-7:
                value, pose = trial, candidate
                break
        else:
            break
    return value, pose


def angular(a, b, axial=False):
    if axial:
        return np.arccos(np.clip(a[:3, 2] @ b[:3, 2], -1., 1.))
    return Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()


def radial_translation(xyz, cloud, seed, plane):
    """Project an axial model onto the radius-height plane before fitting."""
    center = np.mean(np.quantile(xyz[:, :2], (.005, .995), axis=0), axis=0)
    radii = np.column_stack((np.linalg.norm(xyz[:, :2]-center, axis=1), xyz[:, 2]))
    tree = cKDTree(radii)
    def correspond(pose):
        local = (cloud-pose[:3, 3]) @ pose[:3, :3]
        offset = local[:, :2]-center
        radial = np.linalg.norm(offset, axis=1)
        distance, index = tree.query(np.column_stack((radial, local[:, 2])))
        n = int(.91*len(distance))
        keep = np.argpartition(distance, n)[:n]
        target = tree.data[index[keep]]
        surface = np.column_stack((center+offset[keep]*target[:, :1]/np.maximum(radial[keep, None], 1e-6), target[:, 1]))
        return np.sqrt(np.mean(distance[keep]**2)), (local[keep]-surface) @ pose[:3, :3].T
    starts = [seed]
    if plane is not None and seed[:3, 2] @ plane[0] >= np.cos(np.deg2rad(8.)):
        locked = seed.copy()
        axis = Rotation.align_vectors([plane[0]], [seed[:3, 2]])[0].as_matrix()
        locked[:3, :3] = axis @ seed[:3, :3]
        starts.append(locked)
    options = []
    for pose in starts:
        score, error = correspond(pose)
        for _ in range(24):
            correction = np.mean(error, axis=0)
            if np.linalg.norm(correction) <= .000008:
                break
            trial = pose.copy()
            trial[:3, 3] += correction
            cost, gap = correspond(trial)
            if cost >= score or np.linalg.norm(trial[:3, 3]-seed[:3, 3]) > .007:
                break
            score, error, pose = cost, gap, trial
        options.append((score, pose))
    return min(options, key=lambda x: x[0])[1]


def register(name, cloud, seed, plane, catalog):
    xyz, normals, tree, coarse, coarse_tree, low, high = catalog[name]
    small = cloud[np.linspace(0, len(cloud)-1, min(210, len(cloud)), dtype=int)]
    axial = name.startswith("falcon") or name in ("Cole_Hardware_Bowl_Scirocco_YellowBlue", "Ecoforms_Cup_B4_SAN")
    starts = [seed]
    if name.startswith("falcon"):
        starts.extend(axial_candidates(small, seed, low, high))
    elif plane is not None:
        starts.extend(upright_candidates(small, seed, low, high, plane))
    ranked = sorted(((pose_error(coarse_tree, small, p), p) for p in starts), key=lambda x: x[0])
    selected = [seed]
    for _, pose in ranked:
        if all(angular(pose, p) >= .18 or np.linalg.norm(pose[:3, 3]-p[:3, 3]) >= .014 for p in selected):
            selected.append(pose)
        if len(selected) >= 9:
            break
    results = []
    for start in selected:
        first = refine(coarse, small, start, iterations=14, trim=.84,
                          max_translation=.060, max_rotation_deg=38.)
        val, fit = point_plane(xyz, normals, tree, cloud, first)
        loss = val+.00010*min(angular(seed, fit, axial), np.pi)
        if plane is not None and not name.startswith("falcon"):
            normal, offset = plane
            below = (fit[:3, 3] @ normal+offset)+np.quantile(xyz @ (fit[:3, :3].T @ normal), .005)
            loss += .20*max(0., -.003-below)
        results.append((loss, val, fit))
    floor = min(v[1] for v in results)
    loss, error, pose = min((v for v in results if v[1] <= floor+.0012), key=lambda v: v[0])
    if axial and not name.startswith("falcon"):
        trial = radial_translation(xyz, cloud, pose, plane)
        residual = pose_error(tree, cloud, trial)
        if residual < error+.00024:
            pose, error = trial, residual
    return pose, dict(trimmed_rmse_m=error, initial_rmse_m=pose_error(tree, cloud, seed), starts=len(results),
                      combined_loss=loss), results


def register_identities(hypotheses, cloud, plane, catalog):
    results = []
    for name, seed, probability in hypotheses:
        pose, evidence, options = register(name, cloud, seed, plane, catalog)
        cost = evidence["combined_loss"]-.00016*np.log(max(probability, 1e-6))
        results.append((cost, name, pose, evidence, options))
    _, name, pose, evidence, options = min(results, key=lambda v: v[0])
    return name, pose, evidence, options
