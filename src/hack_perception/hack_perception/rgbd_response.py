"""Versioned response for the experimental model-pose service."""

import numpy as np

from .rgbd.geometry import table_z_at
from .rgbd_frames import CAMERAS
from .rgbd_labels import CLASS_BY_ID

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


def infer_response(backend, sensors, stamp, pool, tray_of_color=None, *, prepared=None):
    segmentation, plane, trays = prepared if prepared is not None else prepare_scene(backend, sensors, tray_of_color)
    detections = backend.estimate(sensors, pool, segmentation, plane)
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
                                    pose_status="insufficient_depth", T_pelvis_object=None,
                                    center_pelvis_m=None, size_model_m=None, orientation_mode=None,
                                    detector_score=proposal["confidence"], depth_points=None,
                                    fit_rmse_m=None, depth_supported_views=0, instances={CAMERAS[slot]: ident}))
    containers = [dict(id=tray.id, kind="tray", color=tray.color, center_pelvis_m=tray.center.tolist(),
                       yaw_rad=tray.yaw, size_m=tray.size) for tray in trays]
    return response("ok", stamp=stamp, pool=pool, objects=objects,
                    table_z=table_z_at(plane) if plane is not None else None, containers=containers)
