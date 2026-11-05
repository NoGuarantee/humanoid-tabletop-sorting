#!/usr/bin/env python3
"""Replay a sensor-only manifest through the isolated CPU worker used by ROS."""
import argparse
import json
from pathlib import Path
import re
from time import monotonic, sleep

import numpy as np
from PIL import Image

from hack_perception import table_clusters
from hack_perception.rgbd.geometry import table_z_at
from hack_perception.rgbd_frames import CAMERAS
from hack_perception.rgbd_worker import PoseWorker


def receive(worker, expected):
    deadline = monotonic() + 25.
    while monotonic() < deadline:
        item = worker.poll()
        if item is not None:
            if item[0] != expected:
                raise RuntimeError(f"expected worker {expected}, received {item}")
            return item
        if not worker.process.is_alive():
            raise RuntimeError(f"worker exited with code {worker.process.exitcode}")
        sleep(.005)
    raise TimeoutError(f"worker did not return {expected}")


def baseline_seconds(sensor):
    """CPU cost of the legacy clustering kernel in the caller while pose is in flight."""
    start = monotonic()
    z = sensor["depth"].astype(float) / 1000.
    k, t = sensor["K"], sensor["T"]
    v, u = np.indices(z.shape)
    optical = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)
    cloud = optical @ t[:3, :3].T + t[:3, 3]
    cloud[z == 0] = np.nan
    table_clusters.detect(cloud)
    return monotonic() - start


def replay(args, worker, parser):
    receive(worker, "ready")
    for index, line in enumerate((args.sensor / "manifest.jsonl").read_text().splitlines()):
        row = json.loads(line)
        pool = args.pool
        if pool is None and not args.scene_only:
            match = re.search(r"-p([46])-", row["layout_id"])
            if match is None:
                parser.error("use --pool for a layout_id without a supported pool")
            pool = int(match[1])
        views = [row["views"][camera] for camera in CAMERAS]
        stamp = views[0]["stamp_ns"]
        if stamp <= 0 or views[1]["stamp_ns"] != stamp:
            raise ValueError("expected one nonzero stamp across both cameras")
        sensors = []
        for view in views:
            with Image.open(args.sensor / view["rgb"]) as color, Image.open(args.sensor / view["depth_mm"]) as depth:
                sensors.append(dict(rgb=np.array(color.convert("RGB")), depth=np.array(depth, np.uint16),
                                    K=np.asarray(view["K"]), T=np.asarray(view["T_pelvis_camera"])))
        key = index, stamp
        start = monotonic()
        if not worker.submit("scene", key, (sensors, None)):
            raise RuntimeError("worker request queue is full")
        _, _, (plane, trays), scene_s = receive(worker, "scene")
        value = dict(capture_id=row["capture_id"], stamp_ns=stamp, scene_s=scene_s, scene_elapsed_s=monotonic() - start,
                     table_z_m=None if plane is None else table_z_at(plane, *table_clusters.TABLE_REF_XY),
                     trays=[dict(id=t.id, xyz=t.center.tolist(), yaw=t.yaw) for t in trays])
        if not args.scene_only:
            start = monotonic()
            if not worker.submit("pose", key, pool):
                raise RuntimeError("worker request queue is full")
            value["baseline_during_pose_s"] = baseline_seconds(sensors[0])
            _, _, value["response"], value["pose_s"] = receive(worker, "pose")
            value["pose_elapsed_s"] = monotonic() - start
        print(json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, required=True, help="directory containing manifest.jsonl")
    parser.add_argument("--pool", type=int, choices=(4, 6), help="override the pool in layout_id")
    parser.add_argument("--scene-only", action="store_true")
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    worker = PoseWorker(args.model_dir, not args.scene_only, args.torch_threads)
    try:
        replay(args, worker, parser)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
