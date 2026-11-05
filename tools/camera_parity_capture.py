#!/usr/bin/env python3
"""What perception really gets over ROS at several head poses, optionally compared with a native reference.

The accuracy numbers of the perception research (PR #2) come from a collector that reads the simulator directly.
Before porting it, check that the ROS stream carries the same thing. For each head pose this records both RGB-D
cameras and checks: encodings and sizes, frame ids, one K for color and depth, valid depth range, one stamp across
the cameras, and whether /tf answers at the image stamp with the same transform as a moment later.

Frames go to <out>/<pose>_<camera>.npz (rgb, depth, K, T_pelvis_camera, stamp) for an offline replay, the numbers
to <out>/summary.json. Exit code 0 means every check passed.

With --native FILE the poses come from the file, and K and T_pelvis_camera are compared with it:

  {"poses": [{"name": "p0",
              "joints": {"head_pitch_joint": 0.4, "head_yaw_joint": 0.0},
              "cameras": {"cam_head": {"K": [[...3x3...]], "T_pelvis_camera": [[...4x4...]]},
                          "cam_torso": {"K": [[...]], "T_pelvis_camera": [[...]]}}}]}

Only head joints are commanded. Other joints in "joints" are compared with /joint_states, not moved.

With the simulator running, nothing else moving the head (no full launch, it exits when brain publishes done):

  bash tools/remote.sh dev --timeout 600 -- 'source /opt/ros/humble/setup.bash && \
    python3 tools/camera_parity_capture.py --out camera_parity'
  bash tools/remote.sh fetch work/<your name>/camera_parity
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque

import numpy as np

WIDTH, HEIGHT = 640, 360
HFOV_DEG = 86.0  # INTERFACES.md
DEPTH_MIN_MM, DEPTH_MAX_MM = 150, 6000  # INTERFACES.md: anything else is published as 0
CAMERAS = {"cam_head": "camera_head_link", "cam_torso": "camera_body_front_link"}
HEAD_MOTORS = {"head_roll_joint": 1, "head_pitch_joint": 2, "head_yaw_joint": 3}
# Inside the head range of the native test set: yaw -21..18 deg, pitch 5..26 deg.
DEFAULT_POSES = [
    {"name": "center", "joints": {"head_pitch_joint": 0.40, "head_yaw_joint": 0.00}},
    {"name": "left", "joints": {"head_pitch_joint": 0.20, "head_yaw_joint": 0.30}},
    {"name": "right", "joints": {"head_pitch_joint": 0.30, "head_yaw_joint": -0.30}},
]

JOINT_TOL = 5e-4  # rad, about 0.05 mm at the camera; drives overshoot or undershoot, so the loop closes on this
SETTLE_RAD = 1e-4  # rad, joint spread over SETTLE_WINDOW_S of sim time that counts as stopped
SETTLE_WINDOW_S = 1.0
HEAD_SPEED = 0.5  # rad/s
TF_RECHECK_S = 1.0  # sim seconds between the two /tf lookups of one frame
TF_POS_TOL_M = 1e-4  # the same frame must not move: the target accuracy is below 1 mm
TF_ROT_TOL_DEG = 0.02
K_TOL_PX = 0.01
NATIVE_POS_TOL_M = 2e-4
NATIVE_ROT_TOL_DEG = 0.05
NATIVE_JOINT_TOL = 1e-3  # rad


def quat_matrix(x, y, z, w, tx, ty, tz):
    """4x4 transform from a ROS quaternion (xyzw) and a translation."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), tx],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), ty],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), tz],
        [0.0, 0.0, 0.0, 1.0]])


def transform_delta(a, b):
    """(metres, degrees) between two 4x4 transforms."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), math.degrees(math.acos(min(1.0, max(-1.0, cos))))


def image_array(msg, channels, dtype):
    """Rows of a sensor_msgs/Image as an array, honoring step and endianness."""
    row = msg.width * channels * np.dtype(dtype).itemsize
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :row]
    out = raw.reshape(-1).view(np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<"))
    return out.reshape(msg.height, msg.width, channels).squeeze().astype(dtype)


def check_images(color, depth, frame_id):
    """Failures of one color + depth pair, without looking at the pixels."""
    fails = []
    if (color.encoding, color.step) != ("rgb8", WIDTH * 3):
        fails.append(f"color encoding/step {color.encoding}/{color.step}")
    if (depth.encoding, depth.step) != ("16UC1", WIDTH * 2):
        fails.append(f"depth encoding/step {depth.encoding}/{depth.step}")
    for name, msg in (("color", color), ("depth", depth)):
        if (msg.width, msg.height) != (WIDTH, HEIGHT):
            fails.append(f"{name} size {msg.width}x{msg.height}")
        if msg.header.frame_id != frame_id:
            fails.append(f"{name} frame_id {msg.header.frame_id!r}, expected {frame_id!r}")
    return fails


def check_depth(depth):
    """Numbers and failures for a depth image in millimetres, 0 = no data."""
    valid = depth[depth > 0]
    stats = {"zero_fraction": float(np.mean(depth == 0))}
    fails = []
    if valid.size == 0:
        return stats, ["depth has no valid pixels"]
    stats.update(min_mm=int(valid.min()), max_mm=int(valid.max()))
    if stats["min_mm"] < DEPTH_MIN_MM or stats["max_mm"] >= DEPTH_MAX_MM:
        fails.append(f"depth range {stats['min_mm']}..{stats['max_mm']} mm, 0 was expected outside "
                     f"{DEPTH_MIN_MM}..{DEPTH_MAX_MM}")
    return stats, fails


def check_intrinsics(k_color, k_depth):
    """Numbers and failures for the K of both streams. cx, cy are reported as is: the fit depends on them."""
    k_color, k_depth = np.asarray(k_color, float), np.asarray(k_depth, float)
    fails = []
    gap = float(np.abs(k_color - k_depth).max())
    if gap > K_TOL_PX:
        fails.append(f"color and depth K differ by {gap:.4f} px")
    fx_expected = (WIDTH / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    if abs(k_color[0, 0] - fx_expected) > 0.01 * fx_expected:
        fails.append(f"fx {k_color[0, 0]:.3f}, expected about {fx_expected:.3f} from the {HFOV_DEG} deg FOV")
    return {"fx": float(k_color[0, 0]), "fy": float(k_color[1, 1]),
            "cx": float(k_color[0, 2]), "cy": float(k_color[1, 2])}, fails


def compare_native(native_camera, k, t_pelvis_camera):
    """Numbers and failures of one camera against the native reference."""
    k_gap = float(np.abs(np.asarray(native_camera["K"], float) - k).max())
    pos, rot = transform_delta(native_camera["T_pelvis_camera"], t_pelvis_camera)
    fails = []
    if k_gap > K_TOL_PX:
        fails.append(f"K differs from native by {k_gap:.4f} px")
    if pos > NATIVE_POS_TOL_M or rot > NATIVE_ROT_TOL_DEG:
        fails.append(f"T_pelvis_camera differs from native by {pos * 1e3:.3f} mm, {rot:.3f} deg")
    return {"K_max_px": k_gap, "pos_mm": pos * 1e3, "rot_deg": rot}, fails


def load_poses(path):
    if not path:
        return DEFAULT_POSES
    with open(path, encoding="utf-8") as f:
        return json.load(f)["poses"]


def ros_main(args):
    # Imported here so the checks above can be tried without ROS.
    import rclpy
    from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from tf2_ros import Buffer, TransformListener

    def stamp_ns(header):
        return Time.from_msg(header.stamp).nanoseconds

    class Capture(Node):
        def __init__(self):
            super().__init__("camera_parity_capture", parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            self.lock = threading.Lock()
            self.images = {}  # (camera, kind) -> {stamp_ns: Image}
            self.infos = {}  # (camera, kind) -> CameraInfo
            self.joints = {}
            self.head = self.create_publisher(CmdSetMotorPosition, "/head/cmd_pos", 10)
            self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
            for camera in CAMERAS:
                for kind in ("color", "depth"):
                    key = camera, kind
                    self.images[key] = {}
                    self.create_subscription(Image, f"/{camera}/{kind}/image_raw",
                                             lambda msg, k=key: self.on_image(k, msg), qos_profile_sensor_data)
                    self.create_subscription(CameraInfo, f"/{camera}/{kind}/camera_info",
                                             lambda msg, k=key: self.on_info(k, msg), qos_profile_sensor_data)
            self.tf = Buffer()
            self.listener = TransformListener(self.tf, self)

        def on_joints(self, msg):
            with self.lock:
                self.joints = dict(zip(msg.name, msg.position))

        def on_image(self, key, msg):
            with self.lock:
                frames = self.images[key]
                frames[stamp_ns(msg.header)] = msg
                while len(frames) > 8:
                    del frames[min(frames)]

        def on_info(self, key, msg):
            with self.lock:
                self.infos[key] = msg

        def now_s(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def read_joints(self):
            with self.lock:
                return dict(self.joints)

        def command(self, targets):
            self.head.publish(CmdSetMotorPosition(cmds=[
                SetMotorPosition(name=HEAD_MOTORS[joint], pos=float(pos), spd=HEAD_SPEED, cur=1.0)
                for joint, pos in targets.items()]))

        def wait_stopped(self, joints, targets, deadline):
            """Keep commanding `targets` until `joints` hold still for SETTLE_WINDOW_S of sim time."""
            history = deque()
            while time.monotonic() < deadline:
                self.command(targets)
                time.sleep(0.1)
                now, state = self.now_s(), self.read_joints()
                if not all(j in state for j in joints):
                    continue
                history.append((now, [state[j] for j in joints]))
                # Keep one sample older than the window, so a full window always spans SETTLE_WINDOW_S.
                while len(history) > 1 and now - history[1][0] >= SETTLE_WINDOW_S:
                    history.popleft()
                spread = np.ptp([h[1] for h in history], axis=0).max()
                if now - history[0][0] >= SETTLE_WINDOW_S and spread < SETTLE_RAD:
                    return True
            return False

        def go_to(self, goal, deadline):
            """Closed loop on /joint_states: the drives do not land exactly on the command. Returns the error."""
            head = {j: float(p) for j, p in goal.items() if j in HEAD_MOTORS}
            cmd = dict(head)
            errors = {}
            for _ in range(6):
                if not self.wait_stopped(list(head), cmd, deadline):
                    raise TimeoutError(f"head did not stop on the way to {head}")
                state = self.read_joints()
                errors = {j: head[j] - state[j] for j in head}
                if max(abs(e) for e in errors.values()) < JOINT_TOL:
                    break
                cmd = {j: cmd[j] + errors[j] for j in head}
            return errors, cmd

        def wait_frames(self, after_s, deadline):
            """Newest color + depth per camera taken after `after_s`, preferring one stamp for all cameras."""
            after_ns = int(after_s * 1e9)
            while time.monotonic() < deadline:
                with self.lock:
                    per_camera = {}
                    for camera in CAMERAS:
                        color, depth = self.images[camera, "color"], self.images[camera, "depth"]
                        per_camera[camera] = {s for s in color.keys() & depth.keys() if s > after_ns}
                    if all(per_camera.values()):
                        common = set.intersection(*per_camera.values())
                        stamps = ({c: max(common) for c in CAMERAS} if common
                                  else {c: max(s) for c, s in per_camera.items()})
                        return {c: (self.images[c, "color"][s], self.images[c, "depth"][s])
                                for c, s in stamps.items()}
                time.sleep(0.05)
            raise TimeoutError("no fresh color + depth pair from every camera")

        def lookup(self, frame_id, stamp):
            """T_pelvis_frame at `stamp`, or None and the reason."""
            try:
                msg = self.tf.lookup_transform("pelvis", frame_id, Time.from_msg(stamp))
            except Exception as e:  # tf2 has several exception types, all of them mean the same here
                return None, f"{type(e).__name__}: {e}"
            t, q = msg.transform.translation, msg.transform.rotation
            return quat_matrix(q.x, q.y, q.z, q.w, t.x, t.y, t.z), None

    def capture_pose(node, pose):
        name = pose["name"]
        deadline = time.monotonic() + args.pose_timeout
        report = {"name": name, "fails": []}
        errors, cmd = node.go_to(pose["joints"], deadline)
        report["joint_error_rad"] = errors
        if max(abs(e) for e in errors.values()) >= JOINT_TOL:
            report["fails"].append(f"head stopped off target: {errors}")
        frames = node.wait_frames(node.now_s(), deadline)
        report["stamp_gap_s"] = (max(stamp_ns(c.header) for c, _ in frames.values())
                                 - min(stamp_ns(c.header) for c, _ in frames.values())) * 1e-9
        if report["stamp_gap_s"] > 0:
            report["fails"].append(f"cameras have different stamps, gap {report['stamp_gap_s']:.4f} s")
        state = node.read_joints()
        report["joints"] = state
        for joint, want in pose["joints"].items():
            if joint in state and abs(state[joint] - want) > NATIVE_JOINT_TOL:
                report["fails"].append(f"{joint} is {state[joint]:.4f}, the pose wants {want:.4f}")
            elif joint not in state:
                report["fails"].append(f"{joint} is missing from /joint_states")

        # The second /tf lookup needs the head still; it stays commanded while the sim clock runs.
        recheck_at = node.now_s() + TF_RECHECK_S
        while node.now_s() < recheck_at and time.monotonic() < deadline:
            node.command(cmd)
            time.sleep(0.1)

        report["cameras"] = {}
        for camera, (color_msg, depth_msg) in frames.items():
            fails, frame_id = [], CAMERAS[camera]
            fails += check_images(color_msg, depth_msg, frame_id)
            if stamp_ns(color_msg.header) != stamp_ns(depth_msg.header):
                fails.append("color and depth have different stamps")
            with node.lock:
                infos = node.infos.get((camera, "color")), node.infos.get((camera, "depth"))
            entry = {"stamp_ns": stamp_ns(color_msg.header), "fails": fails}
            report["cameras"][camera] = entry
            if None in infos:
                fails.append("no camera_info")
                continue
            k = np.array(infos[0].k, float).reshape(3, 3)
            entry["intrinsics"], more = check_intrinsics(k, np.array(infos[1].k, float).reshape(3, 3))
            fails += more
            entry["distortion"] = list(infos[0].d)
            if any(abs(c) > 1e-9 for c in infos[0].d):
                fails.append(f"non-zero distortion {list(infos[0].d)}")
            depth = image_array(depth_msg, 1, np.uint16)
            entry["depth"], more = check_depth(depth)
            fails += more

            transform, reason = node.lookup(frame_id, color_msg.header.stamp)
            if transform is None:
                fails.append(f"/tf at the image stamp: {reason}")
                continue
            later, reason = node.lookup(frame_id, Time(seconds=0).to_msg())  # newest available
            if later is None:
                fails.append(f"/tf newest: {reason}")
            else:
                pos, rot = transform_delta(transform, later)
                entry["tf_drift"] = {"pos_mm": pos * 1e3, "rot_deg": rot}
                if pos > TF_POS_TOL_M or rot > TF_ROT_TOL_DEG:
                    fails.append(f"/tf at the stamp and newest differ by {pos * 1e3:.3f} mm, {rot:.3f} deg")
            entry["T_pelvis_camera"] = transform.tolist()
            if "cameras" in pose:
                entry["native"], more = compare_native(pose["cameras"][camera], k, transform)
                fails += more
            np.savez_compressed(os.path.join(args.out, f"{name}_{camera}.npz"),
                                rgb=image_array(color_msg, 3, np.uint8), depth=depth, K=k,
                                T_pelvis_camera=transform, stamp_ns=entry["stamp_ns"])
        return report

    def mission(node, result):
        try:
            poses = load_poses(args.native)
            start = time.monotonic()
            while not node.read_joints() and time.monotonic() - start < 30:
                time.sleep(0.2)
            if not node.read_joints():
                raise TimeoutError("no /joint_states: is the simulator running?")
            time.sleep(args.tf_warmup)  # let the tf buffer fill
            result["poses"] = []
            for pose in poses:
                report = capture_pose(node, pose)
                result["poses"].append(report)
                print_pose(report)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    rclpy.init()
    node = Capture()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    result = {}
    mission(node, result)
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    return result


def all_fails(result):
    out = [f"run: {result['error']}"] if "error" in result else []
    for pose in result.get("poses", []):
        out += [f"{pose['name']}: {f}" for f in pose["fails"]]
        for camera, entry in pose["cameras"].items():
            out += [f"{pose['name']}/{camera}: {f}" for f in entry["fails"]]
    return out


def print_pose(report):
    print(f"--- {report['name']}: joint error {report['joint_error_rad']}, stamp gap {report['stamp_gap_s']:.4f} s")
    for camera, entry in report["cameras"].items():
        print(f"  {camera}: K {entry.get('intrinsics')}, depth {entry.get('depth')}, "
              f"tf drift {entry.get('tf_drift')}, native {entry.get('native')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="camera_parity", help="directory for frames and summary.json")
    parser.add_argument("--native", help="JSON with poses and native K, T_pelvis_camera")
    parser.add_argument("--pose-timeout", type=float, default=120.0, help="wall seconds per head pose")
    parser.add_argument("--tf-warmup", type=float, default=3.0, help="wall seconds before the first pose")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    result = ros_main(args)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    fails = all_fails(result)
    for line in fails:
        print("FAIL", line)
    print(f"{'FAILED' if fails else 'OK'}: {len(result.get('poses', []))} poses, {len(fails)} failures, "
          f"see {args.out}/summary.json")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
