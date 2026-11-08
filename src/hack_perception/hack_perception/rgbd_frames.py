"""Match official registered RGB-D and CameraInfo at a single exposure."""
from collections import OrderedDict

import numpy as np

CAMERAS = ("cam_head", "cam_torso")
STREAMS = {
    "color": "color/image_raw",
    "depth": "depth/image_raw",
    "color_info": "color/camera_info",
    "depth_info": "depth/camera_info",
}


def stamp_ns(header):
    return header.stamp.sec * 10**9 + header.stamp.nanosec


class PairBuffer:
    """Bounded per-topic buffers. The caller serializes reads and writes."""

    def __init__(self, capacity=6):
        self.capacity = capacity
        self.frames = {(camera, kind): OrderedDict() for camera in CAMERAS for kind in STREAMS}

    def add(self, camera, kind, msg):
        frames = self.frames[camera, kind]
        ns = stamp_ns(msg.header)
        frames[ns] = msg
        frames.move_to_end(ns)
        while len(frames) > self.capacity:
            frames.popitem(last=False)

    def clear(self):
        for frames in self.frames.values():
            frames.clear()

    def latest(self, now_ns, max_age_ns, *, paired=True):
        exposures = [set.intersection(*(set(self.frames[camera, kind]) for kind in STREAMS))
                     for camera in CAMERAS]
        candidates = exposures[0] & exposures[1] if paired else exposures[0] | exposures[1]
        ns = max((ns for ns in candidates if ns > 0 and 0 <= now_ns - ns <= max_age_ns), default=None)
        if ns is None:
            return None
        return ns, self.at(ns)

    def at(self, ns):
        views = []
        for camera in CAMERAS:
            view = {kind: self.frames[camera, kind].get(ns) for kind in STREAMS}
            views.append(None if any(msg is None for msg in view.values()) else view)
        return views

    def latest_for(self, camera, now_ns, max_age_ns, *, after_ns=0):
        exposures = set.intersection(*(set(self.frames[camera, kind]) for kind in STREAMS))
        stamp = max((stamp for stamp in exposures if stamp > 0 and stamp >= after_ns
                     and 0 <= now_ns-stamp <= max_age_ns), default=None)
        return None if stamp is None else self.at(stamp)[CAMERAS.index(camera)]


def decode_image(msg, encoding, dtype, channels):
    if msg.encoding != encoding:
        raise ValueError(f"expected {encoding}, got {msg.encoding}")
    pixel_bytes = np.dtype(dtype).itemsize * channels
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * pixel_bytes:
        raise ValueError("invalid image dimensions or row step")
    data = bytes(msg.data)
    if len(data) != msg.step * msg.height:
        raise ValueError("invalid image payload length")
    shape = (msg.height, msg.width, channels)
    strides = (msg.step, pixel_bytes, np.dtype(dtype).itemsize)
    return np.ndarray(shape, dtype=dtype, buffer=data, strides=strides)


def decode_view(frames):
    """Decode the registered stream without changing pixels or depth units."""
    color, depth = frames["color"], frames["depth"]
    frame = depth.header.frame_id
    ns = stamp_ns(depth.header)
    if not frame or ns <= 0 or any(m.header.frame_id != frame or stamp_ns(m.header) != ns for m in frames.values()):
        raise ValueError("camera frame IDs or exposure timestamps differ")
    if any((m.height, m.width) != (depth.height, depth.width) for m in frames.values()):
        raise ValueError("RGB, depth and CameraInfo dimensions differ")
    color_info, depth_info = frames["color_info"], frames["depth_info"]
    k = np.asarray(color_info.k, dtype=float).reshape(3, 3)
    kd = np.asarray(depth_info.k, dtype=float).reshape(3, 3)
    if (not np.isfinite(k).all() or not np.isfinite(kd).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose(k, kd, rtol=0, atol=1e-6)):
        raise ValueError("RGB and depth must have matching pinhole intrinsics")
    if any(not np.allclose(m.d, 0, rtol=0, atol=1e-9) for m in (color_info, depth_info)):
        raise ValueError("RGB and depth must be undistorted")
    rgb = decode_image(color, "rgb8", np.uint8, 3)
    dtype = ">u2" if depth.is_bigendian else "<u2"
    depth_mm = decode_image(depth, "16UC1", dtype, 1)[..., 0].astype(np.uint16, copy=False)
    return {"rgb": rgb, "depth": depth_mm, "K": k}


def decode_pair(frames, transform_at):
    headers = [view["depth"].header for view in frames]
    if len({h.frame_id for h in headers}) != len(CAMERAS) or len({stamp_ns(h) for h in headers}) != 1:
        raise ValueError("expected two distinct optical frames at the same exposure")
    sensors = [decode_view(view) for view in frames]
    for sensor, header in zip(sensors, headers):
        sensor["T"] = transform_at(header)
    return sensors
