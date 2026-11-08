"""Object-centred crops and pose features from two registered RGB-D views."""
import numpy as np

from .geometry import depth_mask_to_pelvis

SIZE = 64
SCALE_M = .15


def make_features(observations, category):
    """RGB, relative XYZ, object/valid masks and camera axes in two fixed slots.

    Each observation is None or carries registered RGB, depth, mask, K and TF.
    The anchor is the visible depth median in pelvis.
    """
    if len(observations) != 2 or category not in (5, 6):
        raise ValueError("two cameras and Falcon class required")
    clouds, usable = [], []
    for obs in observations:
        if obs is None:
            usable.append(None)
            continue
        depth, mask = np.asarray(obs["depth"]), np.asarray(obs["mask"])
        points = depth_mask_to_pelvis(depth, mask, obs["K"], obs["T"])
        if len(points) < 30:
            usable.append(None)
            continue
        if len(points) > 600:
            points = points[np.linspace(0, len(points) - 1, 600, dtype=int)]
        clouds.append(points)
        usable.append(obs)
    if not clouds:
        return None
    anchor = np.median(np.concatenate(clouds), axis=0)
    images = np.zeros((2, 8, SIZE, SIZE), dtype=np.float32)
    axes = np.zeros((2, 3), np.float32)
    presence = np.zeros(2, np.float32)
    for i, obs in enumerate(usable):
        if obs is None:
            continue
        rgb = np.asarray(obs["rgb"], np.uint8)
        depth = np.asarray(obs["depth"], np.uint16)
        mask = np.asarray(obs["mask"], bool)
        if rgb.shape != depth.shape + (3,) or mask.shape != depth.shape:
            raise ValueError("registered RGB-D-mask shapes required")
        rows, columns = np.where(mask)
        if len(columns) < 1:
            continue
        pad = max(4, int(max(columns.max() - columns.min() + 1, rows.max() - rows.min() + 1) * .15))
        y0 = max(0, int(rows.min()) - pad)
        y1 = min(depth.shape[0], int(rows.max()) + pad + 1)
        x0 = max(0, int(columns.min()) - pad)
        x1 = min(depth.shape[1], int(columns.max()) + pad + 1)
        ys = np.rint(np.linspace(y0, y1 - 1, SIZE)).astype(int)
        xs = np.rint(np.linspace(x0, x1 - 1, SIZE)).astype(int)
        y, x = np.meshgrid(ys, xs, indexing="ij")
        z = depth[y, x].astype(float) / 1000
        cropped_mask = mask[y, x]
        valid = cropped_mask & (z >= .15) & (z <= 6.)
        k, transform = np.asarray(obs["K"], float), np.asarray(obs["T"], float)
        rays = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(k).T
        points = (rays * (z / rays[..., 2])[..., None]) @ transform[:3, :3].T + transform[:3, 3]
        xyz = np.where(valid[..., None], np.clip((points - anchor) / SCALE_M, -2, 2), 0)
        pixels = (rgb[y, x].astype(np.float32) / 255) * cropped_mask[..., None]
        images[i, :3] = pixels.transpose(2, 0, 1)
        images[i, 3:6] = xyz.transpose(2, 0, 1)
        images[i, 6] = cropped_mask
        images[i, 7] = valid
        axes[i] = transform[:3, 2]
        presence[i] = 1.
    extra = np.concatenate((axes.ravel(), presence, np.array([category == 5, category == 6], np.float32),
                            np.clip(anchor / .8, -2, 2))).astype(np.float32)
    if not np.isfinite(images).all() or not np.isfinite(extra).all():
        raise ValueError("invalid features")
    return images, extra, anchor
