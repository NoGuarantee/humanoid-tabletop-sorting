"""Remove measured robot depth points using robot-only spheres and visual hulls.

This is a model-based exclusion, not evidence of free space or object identity.
Objects touching the robot may lose some pixels and must remain unknown. The
caller must align the measured joints and optical camera transform to exposure.
"""

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps([value.dtype.str, value.shape]).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _calibration(K, T):
    k, transform = np.asarray(K, dtype=float), np.asarray(T, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-12)
            or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError('expected finite invertible pinhole calibration')
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0, atol=1e-5)
            or abs(np.linalg.det(transform[:3, :3])-1) > 1e-5):
        raise ValueError('expected rigid optical camera to pelvis transform')
    return k, transform


def _pixel_roi(center, radius, k, height, width):
    """Conservative projection of a sphere, including near-plane crossings."""
    z = center[2]
    if z+radius <= 0:
        return None
    if z-radius <= 0:
        return 0, height, 0, width
    denominator = z*z-radius*radius
    intervals = []
    for component in center[:2]:
        spread = radius*math.sqrt(max(0., component*component+denominator))
        intervals.append(((component*z-spread)/denominator,
                          (component*z+spread)/denominator))
    bounds = []
    for row in k[:2]:
        lo = hi = row[2]
        for coefficient, interval in zip(row[:2], intervals):
            values = coefficient*np.asarray(interval)
            lo += min(values)
            hi += max(values)
        bounds.append((lo, hi))
    # One extra pixel protects projection rounding; the exact 3D predicate
    # below is the only operation that can mark a measured point as robot.
    x0, x1 = max(0, math.floor(bounds[0][0])-1), min(width, math.ceil(bounds[0][1])+2)
    y0, y1 = max(0, math.floor(bounds[1][0])-1), min(height, math.ceil(bounds[1][1])+2)
    return None if x0 >= x1 or y0 >= y1 else (y0, y1, x0, x1)


def mask_robot_depth(depth_m, K, T, q, geometry, margin_m=.003):
    """Return ``{mask, provenance}`` without changing any input.

    ``depth_m`` is a floating HxW optical-Z image in metres. K and T follow the
    existing RGB-D packet: T maps optical camera coordinates into pelvis. Pixel
    rays use integer (column, row) coordinates, as ``depth_mask_to_pelvis`` does.
    Nonpositive/nonfinite depth is unknown and is never marked as robot. Every
    ``geometry.required`` joint must be explicitly measured, finite and numeric.

    Spheres are projected to conservative pixel ROIs; backprojection occurs
    once per image, not once per sphere. Only points inside the model sphere
    expanded by the explicit margin are masked, so front/behind depth survives.
    """
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or depth.dtype.kind != 'f' or not all(depth.shape):
        raise ValueError('expected nonempty floating optical-Z depth in metres')
    if isinstance(margin_m, (bool, np.bool_)) or not isinstance(margin_m, Real):
        raise ValueError('expected finite robot model margin')
    margin = float(margin_m)
    if not math.isfinite(margin) or not 0 <= margin <= .02:
        raise ValueError('robot model margin must be within [0, .02] metres')
    k, transform = _calibration(K, T)
    if not isinstance(q, Mapping):
        raise ValueError('expected measured joint mapping')
    required = tuple(geometry.required)
    if not required or any(name not in q for name in required):
        raise ValueError('missing measured robot joints')
    positions = {}
    for name, value in q.items():
        if (not isinstance(name, str) or isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real) or not math.isfinite(float(value))):
            raise ValueError('expected finite numeric measured robot joints')
        positions[name] = float(value)

    spheres = list(geometry.spheres(positions))
    if not spheres:
        raise ValueError('empty robot sphere model')
    centers, radii, names = [], [], []
    for name, point, radius in spheres:
        point = np.asarray(point, dtype=float)
        if (not isinstance(name, str) or not name or point.shape != (3,)
                or not np.isfinite(point).all() or not math.isfinite(float(radius)) or radius <= 0):
            raise ValueError('invalid robot sphere model')
        names.append(name)
        centers.append(point)
        radii.append(float(radius))
    centers, radii = np.asarray(centers), np.asarray(radii)
    camera_centers = (centers-transform[:3, 3]) @ transform[:3, :3]
    valid = np.isfinite(depth) & (depth > 0)
    mask = np.zeros(depth.shape, dtype=bool)
    height, width = depth.shape
    # Invalid depth is never used in geometry arithmetic (and remains unknown).
    z = np.where(valid, depth, 0.).astype(float)
    inverse = np.linalg.inv(k)
    x, y = np.arange(width)[None, :], np.arange(height)[:, None]
    optical = np.empty((*depth.shape, 3), dtype=float)
    optical[..., 0] = z*(inverse[0, 0]*x+inverse[0, 1]*y+inverse[0, 2])
    optical[..., 1] = z*(inverse[1, 0]*x+inverse[1, 1]*y+inverse[1, 2])
    optical[..., 2] = z
    roi_pixels = tested_pixels = projected_spheres = 0
    if valid.any():
        minimum, maximum = float(z[valid].min()), float(z[valid].max())
        for center, raw_radius in zip(camera_centers, radii):
            radius = raw_radius+margin
            if center[2]+radius < minimum or center[2]-radius > maximum:
                continue
            roi = _pixel_roi(center, radius, k, height, width)
            if roi is None:
                continue
            y0, y1, x0, x1 = roi
            projected_spheres += 1
            roi_pixels += (y1-y0)*(x1-x0)
            pending = valid[y0:y1, x0:x1] & ~mask[y0:y1, x0:x1]
            iy, ix = np.where(pending)
            tested_pixels += len(iy)
            delta = optical[y0:y1, x0:x1][iy, ix]-center
            inside = np.einsum('ij,ij->i', delta, delta) <= radius*radius
            mask[y0+iy[inside], x0+ix[inside]] = True

    body_model=getattr(geometry,'body_visual_hulls',None)
    body_provenance=None
    if body_model is not None:
        body_mask,body_provenance=body_model.mask_optical(optical,k,transform,positions,
            valid & ~mask,margin,_pixel_roi)
        mask |= body_mask
    model_files = {name: hashlib.sha256(Path(getattr(geometry, attribute)).read_bytes()).hexdigest()
                   for name, attribute in (('urdf', 'urdf_path'), ('curobo_spheres', 'cfg_path'),
                                           ('native_hand_spheres', 'hand_path'))}
    q_hash = hashlib.sha256(json.dumps(positions, sort_keys=True, allow_nan=False).encode()).hexdigest()
    sphere_hash = hashlib.sha256(json.dumps(names).encode()+np.column_stack((centers, radii)).tobytes()).hexdigest()
    provenance=dict(
        schema='hack_perception.robot_self_mask/v1',
        method='measured_depth_points_inside_robot_spheres/v1', frame_id='pelvis',
        depth_convention='optical_z_metres_integer_pixel_coordinates', margin_m=margin,
        robot_files_sha256=model_files, sphere_geometry_sha256=sphere_hash, q_sha256=q_hash,
        depth_sha256=_array_hash(depth), K_sha256=_array_hash(k), T_sha256=_array_hash(transform),
        mask_sha256=_array_hash(mask), implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        robot_sphere_count=len(spheres), projected_sphere_count=projected_spheres,
        candidate_roi_pixels=roi_pixels, tested_pixels=tested_pixels,
        valid_depth_pixels=int(valid.sum()), masked_pixels=int(mask.sum()),
        limitations=['Robot spheres and the explicit margin approximate the robot surface.',
                     'Objects touching the robot may be partly masked; exclusion is not positive object or free-space evidence.',
                     'The caller must prove measured joint and camera exposure alignment.'])
    if body_provenance is not None:
        provenance['method']='measured_depth_points_inside_robot_spheres_and_native_visual_hulls/v1'
        provenance['robot_files_sha256']['native_body_visual_hulls']=body_provenance['asset_sha256']
        provenance['body_visual_hulls']=body_provenance
    return dict(mask=mask,provenance=provenance)
