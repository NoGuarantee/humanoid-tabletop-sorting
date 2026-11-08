#!/usr/bin/env python3
"""Record cam_head and cam_torso colour into one side-by-side mp4, paced by /clock.

Run inside the dev container (ROS 2 Humble, python3-opencv), not on the laptop:

  python3 .cursor/skills/sim-video/scripts/record_cameras.py --out log/video/run.mp4 --until-done

One second of video is one second of sim time. SIGTERM closes the file, so the dev
timeout still leaves a playable mp4. Exit 0 prints `wrote ...`.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

CAMERAS = {"head": "/cam_head/color/image_raw", "torso": "/cam_torso/color/image_raw"}
BAR = 24
WAIT_FRAME_S = 90.0
WAIT_CLOCK_S = 30.0


def to_bgr(msg):
    """sensor_msgs/Image rgb8 or bgr8 -> HxWx3 uint8 BGR, honoring step."""
    height, width = int(msg.height), int(msg.width)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    row = raw.reshape(height, int(msg.step))[:, : width * 3].reshape(height, width, 3)
    if msg.encoding == "rgb8":
        return row[:, :, ::-1].copy()
    if msg.encoding == "bgr8":
        return row.copy()
    raise RuntimeError(f"unsupported encoding {msg.encoding}")


def panel(frame, label, width, height):
    import cv2

    canvas = np.zeros((height + BAR, width, 3), np.uint8)
    if frame is None:
        cv2.putText(canvas, "no frame", (8, BAR + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (180, 180, 180), 1, cv2.LINE_AA)
    else:
        img = frame
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        canvas[BAR:, :] = img
    cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def open_writer(path, fps, size):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v is what opencv in the sim image writes. avc1 is often not built in.
    for fourcc, dest in (("mp4v", path), ("XVID", path.with_suffix(".avi"))):
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, dest
        writer.release()
    raise RuntimeError(f"VideoWriter failed for {path} ({size[0]}x{size[1]} @ {fps})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="", help="mp4 path; default log/video/cameras-<wall time>.mp4")
    parser.add_argument("--cameras", default="head,torso", help="comma list: head, torso")
    parser.add_argument("--fps", type=float, default=10.0, help="video frames per sim second")
    parser.add_argument("--seconds", type=float, default=600.0, help="stop after this much sim time")
    parser.add_argument("--until-done", action="store_true", help="also stop when /sim/task/done arrives")
    args = parser.parse_args()
    if args.fps <= 0 or args.seconds <= 0:
        print("fps and seconds must be positive", file=sys.stderr)
        return 1
    names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    unknown = [c for c in names if c not in CAMERAS]
    if not names or unknown:
        print(f"cameras must be from {list(CAMERAS)}, got {unknown or names}", file=sys.stderr)
        return 1
    out = args.out or time.strftime("log/video/cameras-%Y%m%d-%H%M%S.mp4")

    try:
        import cv2
    except ImportError:
        print("no cv2: run this in the dev container, not on the laptop", file=sys.stderr)
        return 1

    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Empty

    stop = False

    def on_signal(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    rclpy.init()
    node = Node("sim_video", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    latest = {name: None for name in names}
    done = False

    def on_image(name):
        def callback(msg):
            try:
                latest[name] = to_bgr(msg)
            except Exception as exc:  # noqa: BLE001 — keep the other camera
                print(f"{name}: {exc}", file=sys.stderr)
        return callback

    for name in names:
        node.create_subscription(Image, CAMERAS[name], on_image(name), qos)

    def on_done(_msg):
        nonlocal done
        done = True

    if args.until_done:
        node.create_subscription(Empty, "/sim/task/done", on_done, 10)

    writer = None
    dest = None
    frames = 0
    origin = None
    next_write = None
    width = height = 0
    period = 1.0 / args.fps
    wall0 = time.monotonic()
    now = 0.0
    rc = 0
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = node.get_clock().now().nanoseconds / 1e9
            if not any(frame is not None for frame in latest.values()):
                if time.monotonic() - wall0 > WAIT_FRAME_S:
                    print("no camera frames in 90 s wall; simulator up, and this process in the dev container?",
                          file=sys.stderr)
                    rc = 1
                    break
                continue
            if now <= 0.0:
                if time.monotonic() - wall0 > WAIT_CLOCK_S:
                    print("cameras publish, but /clock is not advancing (use_sim_time)", file=sys.stderr)
                    rc = 1
                    break
                continue
            if writer is None:
                sample = next(frame for frame in latest.values() if frame is not None)
                height, width = sample.shape[:2]
                writer, dest = open_writer(out, args.fps, (width * len(names), height + BAR))
                print(f"recording {dest.resolve()}", flush=True)
                origin = now
                next_write = now
            if now + 1.0 < next_write:  # sim time jumped backwards: a new layout
                origin = now
                next_write = now
            if now + 1e-4 < next_write:
                if args.until_done and done and frames > 0:
                    break
                continue
            tiles = [panel(latest[name], name, width, height) for name in names]
            writer.write(cv2.hconcat(tiles))
            frames += 1
            next_write += period
            if next_write < now - 2 * period:  # stalled; do not dump a burst of the same frame
                next_write = now
            if args.until_done and done:
                break
            if origin is not None and now - origin >= args.seconds:
                break
    finally:
        if writer is not None:
            writer.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if rc != 0:
        return rc
    if frames == 0:
        print("stopped before any frame was written", file=sys.stderr)
        return 1
    span = 0.0 if origin is None else max(0.0, now - origin)
    print(f"wrote {frames} frames, {span:.1f} sim s -> {dest.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
