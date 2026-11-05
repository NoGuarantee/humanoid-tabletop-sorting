#!/usr/bin/env python3
"""Run the ROS model-pose backend on a sensor-only RGB-D manifest from PR #2."""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from hack_perception.rgbd.pose_backend import PoseBackend
from hack_perception.rgbd_frames import CAMERAS
from hack_perception.rgbd_response import infer_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, required=True, help="directory containing manifest.jsonl")
    parser.add_argument("--pool", type=int, choices=(4, 6), help="override the pool in layout_id")
    args = parser.parse_args()
    backend = PoseBackend(args.model_dir)
    for line in (args.sensor / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        pool = args.pool
        if pool is None:
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
        start = time.monotonic()
        payload = infer_response(backend, sensors, stamp, pool)
        print(json.dumps(dict(capture_id=row["capture_id"], seconds=time.monotonic() - start, response=payload),
                         allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
