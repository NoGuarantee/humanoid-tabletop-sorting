"""Versioned response for the experimental model-pose service."""

import numpy as np

from .rgbd_frames import CAMERAS

SCHEMA = "hack_perception.model_poses/v1"
CATEGORY = {1: "figure", 2: "dish", 3: "box", 4: "electronics", 5: "falcon50", 6: "falcon15"}


def response(status, *, stamp=None, pool=None, objects=(), table_z=None):
    return dict(schema=SCHEMA, status=status, stamp_ns=stamp, frame_id="pelvis",
                pool=pool, table_z_m=table_z, objects=list(objects))


def object_response(detection):
    status = "ok"
    if detection.views == 0:
        status = "insufficient_depth"
    elif detection.views == 1:
        status = "reacquire_single_view"
    elif detection.clipped:
        status = "reacquire_clipped"
    elif not detection.valid:
        status = "reacquire_uncertain_fit"
    instance_ids = {camera: int(ident) for camera, ident in zip(CAMERAS, detection.instances) if ident is not None}
    return dict(name=detection.name, category=CATEGORY[detection.category], pose_status=status,
                T_pelvis_object=detection.pose.tolist() if status == "ok" else None,
                center_pelvis_m=detection.center.tolist() if status == "ok" else None,
                size_model_m=detection.size.tolist(), orientation_mode=detection.orientation,
                detector_score=detection.confidence, depth_points=int(detection.points),
                fit_rmse_m=detection.residual if np.isfinite(detection.residual) else None,
                depth_supported_views=detection.views, instances=instance_ids)


def infer_response(backend, sensors, stamp, pool):
    segmentation = [backend.segment(sensor) for sensor in sensors]
    detections, table_z = backend.estimate(sensors, pool, "table", segmentation=segmentation)
    objects = [object_response(d) for d in detections]
    for slot, (_, proposals, _) in enumerate(segmentation):
        seen = {d.instances[slot] for d in detections}
        for proposal in proposals:
            ident = proposal["instance_id"]
            category = proposal["class_id"]
            if category in backend.pool_categories(pool) and ident not in seen:
                objects.append(dict(name=None, category=CATEGORY[category],
                                    pose_status="insufficient_depth", T_pelvis_object=None,
                                    center_pelvis_m=None, size_model_m=None, orientation_mode=None,
                                    detector_score=proposal["confidence"], depth_points=None,
                                    fit_rmse_m=None, depth_supported_views=0, instances={CAMERAS[slot]: ident}))
    return response("ok", stamp=stamp, pool=pool, objects=objects, table_z=table_z)
