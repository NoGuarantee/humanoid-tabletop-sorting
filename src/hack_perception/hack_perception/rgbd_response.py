"""Versioned response for the experimental model-pose service."""

import numpy as np

from .rgbd.geometry import depth_mask_to_pelvis, table_z_at
from .rgbd_frames import CAMERAS
from .rgbd_labels import CLASS_BY_ID, TRAY_CLASS

SCHEMA = "hack_perception.model_poses/v1"


def response(status, *, stamp=None, pool=None, objects=(), table_z=None, containers=()):
    return dict(schema=SCHEMA, status=status, stamp_ns=stamp, frame_id="pelvis",
                pool=pool, table_z_m=table_z, objects=list(objects), containers=list(containers))


def object_response(detection):
    status = "ok"
    if detection.views == 0:
        status = "insufficient_depth"
    elif detection.clipped:
        status = "reacquire_clipped"
    elif not detection.valid:
        status = "reacquire_uncertain_fit"
    instance_ids = {camera: int(ident) for camera, ident in zip(CAMERAS, detection.instances) if ident is not None}
    return dict(name=detection.name, category=CLASS_BY_ID[detection.category], pose_status=status,
                T_pelvis_object=detection.pose.tolist() if status == "ok" else None,
                center_pelvis_m=detection.center.tolist() if status == "ok" else None,
                size_model_m=detection.size.tolist(), orientation_mode=detection.orientation,
                detector_score=detection.confidence, depth_points=int(detection.points),
                fit_rmse_m=detection.residual if np.isfinite(detection.residual) else None,
                depth_supported_views=detection.views, instances=instance_ids)


def prepare_scene(backend, sensors, tray_of_color):
    segmentation = [backend.segment(sensor) if sensor is not None else None for sensor in sensors]
    plane, trays = backend.scene_geometry(sensors, segmentation=segmentation, tray_of_color=tray_of_color)
    return segmentation, plane, trays


def observed_components(points, plane):
    """Split disconnected mask surfaces without discarding unknown obstacles.

    Connectivity and the primary support-component rule follow grasp-worktree
    b88a154. Unlike that filter, retain every other component: height or size
    alone does not establish that a surface belongs to the robot. Only the
    primary component inherits the source mask identity.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if not len(points):
        return []
    if len(points) < 30:
        samples, inverse = points, np.arange(len(points))
    else:
        # Preserve the pre-existing connectivity policy for substantial clouds.
        _, first, inverse = np.unique(np.floor(points / .003).astype(np.int64), axis=0,
                                      return_index=True, return_inverse=True)
        samples = points[first]
    pairs = cKDTree(samples).query_pairs(.012, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(samples), len(samples)))
    _, labels = connected_components(graph, directed=False)
    labels = labels[inverse]
    substantial = np.flatnonzero(np.bincount(labels) >= 30)
    order = np.argsort(labels, kind='stable')
    all_groups = np.split(points[order], np.flatnonzero(np.diff(labels[order]))+1)
    if not len(substantial):
        return all_groups  # Every disconnected sparse surface stays separate.
    groups = [all_groups[label] for label in substantial]
    if plane is None:
        primary = max(range(len(groups)), key=lambda index: len(groups[index]))
    else:
        normal, offset = plane
        heights = [float(np.quantile(group @ normal + offset, .02)) for group in groups]
        lowest = min(heights)
        primary = max((index for index, height in enumerate(heights) if height <= lowest + .02),
                      key=lambda index: len(groups[index]))
    ordered = [groups[primary], *(group for index, group in enumerate(groups) if index != primary)]
    # Combining disconnected fragments invents an occupied bridge through empty
    # space. Retain every fragment with its own bounds and source provenance.
    substantial_set = set(substantial)
    ordered.extend(group for label, group in enumerate(all_groups) if label not in substantial_set)
    return ordered


def observed_regions(sensors, segmentation, plane, categories=None):
    """Visible depth bounds for nearby instances; these are not complete object models."""
    regions = []
    for camera, sensor, segment in zip(CAMERAS, sensors, segmentation):
        if sensor is None or segment is None:
            continue
        raster, proposals, _ = segment
        source_ids = {proposal['instance_id'] for proposal in proposals}
        extra_id = -1
        for proposal in proposals:
            # A pose/identity filter is never an obstacle filter. Trays and
            # racks have separate geometry; retain all other observed masks.
            if proposal['class_id'] in (TRAY_CLASS, 8):
                continue
            points = depth_mask_to_pelvis(sensor['depth'], raster == proposal['instance_id'],
                                          sensor['K'], sensor['T'])
            if plane is not None:
                normal, offset = plane
                points = points[points @ normal + offset > .006]
            if not len(points):
                continue
            # Height alone cannot identify a robot hand. Unknown high surfaces
            # and sparse fragments remain obstacles. These are visible bounds,
            # not a reconstruction of occluded geometry.
            for index, component in enumerate(observed_components(points, plane)):
                primary = index == 0 and len(component) >= 30
                # A sparse curved surface can have an enclosing box that
                # intersects the robot although every measured point clears it.
                # Keep each point, including duplicates; the consumer inflates
                # only zero-width axes using its existing obstacle policy.
                fragments = component[:, None, :] if len(component) < 30 else (component,)
                for fragment in fragments:
                    low, high = fragment.min(axis=0), fragment.max(axis=0)
                    ident = proposal['instance_id']
                    if not primary:
                        while extra_id in source_ids:
                            extra_id -= 1
                        ident = extra_id
                        extra_id -= 1
                    region = dict(low_pelvis_m=low.tolist(), high_pelvis_m=high.tolist(),
                                  category=CLASS_BY_ID.get(proposal['class_id'], 'unknown') if primary else 'unknown',
                                  instances={camera: ident}, depth_points=len(fragment))
                    if not primary:
                        region['source_instances'] = {camera: proposal['instance_id']}
                    regions.append(region)
    return regions


def infer_response(backend, sensors, stamp, pool, tray_of_color=None, *, prepared=None, names=None):
    segmentation, plane, trays = prepared if prepared is not None else prepare_scene(backend, sensors, tray_of_color)
    detections = (backend.estimate(sensors, pool, segmentation, plane, names=names) if names else
                  backend.estimate(sensors, pool, segmentation, plane))
    objects = [object_response(d) for d in detections]
    for slot, camera_segmentation in enumerate(segmentation):
        if camera_segmentation is None:
            continue
        _, proposals, _ = camera_segmentation
        seen = {d.instances[slot] for d in detections}
        for proposal in proposals:
            ident = proposal["instance_id"]
            category = proposal["class_id"]
            if category in backend.pool_categories(pool) and ident not in seen:
                objects.append(dict(name=None, category=CLASS_BY_ID[category],
                                    pose_status="not_fitted" if names else "insufficient_depth", T_pelvis_object=None,
                                    center_pelvis_m=None, size_model_m=None, orientation_mode=None,
                                    detector_score=proposal["confidence"], depth_points=None,
                                    fit_rmse_m=None, depth_supported_views=0, instances={CAMERAS[slot]: ident}))
    containers = [dict(id=tray.id, kind="tray", color=tray.color, center_pelvis_m=tray.center.tolist(),
                       yaw_rad=tray.yaw, size_m=tray.size) for tray in trays]
    if plane is not None:
        up = np.asarray(plane[0], dtype=float)
        up = up / np.linalg.norm(up)
        if up[2] < 0:
            up = -up
        for container in containers:
            yaw = container['yaw_rad']
            heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
            heading -= up * np.dot(heading, up)
            heading /= np.linalg.norm(heading)
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack([heading, np.cross(up, heading), up])
            pose[:3, 3] = container['center_pelvis_m']
            container['T_pelvis_opening'] = pose.tolist()
    payload = response("ok", stamp=stamp, pool=pool, objects=objects,
                       table_z=table_z_at(plane) if plane is not None else None, containers=containers)
    if plane is not None:
        payload['table_normal_pelvis'] = up.tolist()
    payload['observed_regions'] = observed_regions(sensors, segmentation, plane)
    return payload


def bind_exposure(payload, key, session_id, cameras):
    """Scope raster IDs to this process, clock/task epoch and sensor exposure.

    Clients compare the entire key AND camera/instance ID before excluding a
    target region. source_instances is provenance only, never target identity.
    """
    exposure = dict(session_id=session_id, epoch=key[0], stamp_ns=key[1])
    payload['exposure_key'] = exposure
    payload['source_cameras'] = list(cameras)
    for item in [*payload['objects'], *payload['containers'], *payload['observed_regions']]:
        item['exposure_key'] = dict(exposure)
    return payload
