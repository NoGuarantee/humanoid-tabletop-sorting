#!/usr/bin/env python3
"""Bounded replay of frozen RGB-D/CameraInfo/TF, without any scene state input."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
from time import perf_counter

os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
import numpy as np
from PIL import Image
import torch
from hack_perception.rgbd.pose_backend import PoseBackend
from hack_perception.rgbd_response import infer_response


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='directory containing inputs.json and sensor PNGs')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='one device per process; Ultralytics mutates CUDA visibility')
    parser.add_argument('--names', default='')
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 5:
        parser.error('repeats must be 2..5, first iteration is warmup')
    torch.set_num_threads(2)
    source = Path(__file__).resolve().parents[1] / 'src/hack_perception'
    hashes = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*.py')) if not p.name.startswith('._')}
    manifest = json.loads((args.data/'inputs.json').read_text())
    inputs = {'inputs.json': digest(args.data/'inputs.json')}
    rows = []
    for device in (args.device,):
        backend = PoseBackend(args.bundle, device=device)
        for record in manifest['captures']:
            sensors = []
            for view in record['views']:
                for key in ('rgb', 'depth_mm'):
                    inputs[view[key]] = digest(args.data/view[key])
                sensors.append(dict(rgb=np.array(Image.open(args.data/view['rgb']).convert('RGB')),
                    depth=np.array(Image.open(args.data/view['depth_mm'])).astype(np.uint16),
                    K=np.array(view['K']), T=np.array(view['T_pelvis_camera'])))
            trials = []
            for _ in range(args.repeats):
                start = perf_counter()
                segmentation = [backend.segment(sensor) for sensor in sensors]
                segmented = perf_counter()
                plane, trays = backend.scene_geometry(sensors, segmentation=segmentation)
                geometry = perf_counter()
                payload = infer_response(backend, sensors, 1, record['pool'],
                    prepared=(segmentation, plane, trays), names=args.names.split(',') if args.names else None)
                finish = perf_counter()
                trials.append(dict(segmentation_s=segmented-start, geometry_s=geometry-segmented,
                    pose_and_regions_s=finish-geometry, total_s=finish-start))
            row = dict(device=device, capture_id=record['capture_id'], trials=trials,
                warm_median_s={key:float(np.median([t[key] for t in trials[1:]])) for key in trials[0]},
                result=payload)
            rows.append(row)
            print(json.dumps({key:value for key,value in row.items() if key not in ('result', 'trials')}), flush=True)
        del backend
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    receipt = dict(scope='saved sensor replay; excludes ROS/DDS and live freshness; not accuracy or robot success',
        torch=torch.__version__, threads=2, names=args.names, source_sha256=hashes, input_sha256=inputs,
        bundle_manifest_sha256=digest(args.bundle/'bundle.json'), results=rows,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
