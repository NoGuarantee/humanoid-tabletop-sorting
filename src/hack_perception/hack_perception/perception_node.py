"""Perception service /perception/perceive. Owner: perception.

Keeps the latest colour, depth and camera_info of both cameras; on request turns them into SceneObjects in `pelvis`.
Runs on request only (not a stream): saves CPU, which counts in tie-breaks.

Status:
  M1: table plane (RANSAC) -> points above it -> clusters -> centre / yaw (PCA) / size / top_z
  M2: trays by colour (red / blue / green), category by size and colour against /sim/objects/names
  M3: YOLO-seg trained on sim frames of all 6 pools
  M4: racks, slots, tube caps and sizes
"""
import json
import math
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from hack_interfaces.msg import Container, DetectedObject, SceneObjects
from hack_interfaces.srv import Perceive
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from . import classify
from . import geometry as geo
from . import table_clusters as tc

CAMERAS = {"head": "cam_head", "torso": "cam_torso"}
BASE = "pelvis"
ARM_FRAMES = (("elbow_pitch_l_link", "wrist_pitch_l_link", "left_tcp_link"),
              ("elbow_pitch_r_link", "wrist_pitch_r_link", "right_tcp_link"))
FINGERS = 0.21  # along -z of the TCP; keep equal to grasp_planner.FINGERTIP


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("hack_perception")
        self.lock = threading.Lock()
        self.frames = {cam: {} for cam in CAMERAS}
        self.tracker = tc.Tracker()
        self.names = {}  # object name -> category, from /sim/objects/names
        self.tray_of_color = dict(classify.DEFAULT_TRAY)  # replaced by /sim/task when it arrives
        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/sim/objects/names", self.on_names, latched, callback_group=cb)
        self.create_subscription(String, "/sim/task", self.on_task, latched, callback_group=cb)
        for cam, topic in CAMERAS.items():
            for kind in ("color", "depth"):
                self.create_subscription(Image, f"/{topic}/{kind}/image_raw",
                                         lambda m, c=cam, k=kind: self.store(c, k, m), 2, callback_group=cb)
            self.create_subscription(CameraInfo, f"/{topic}/color/camera_info",
                                     lambda m, c=cam: self.store(c, "info", m), 1, callback_group=cb)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self, spin_thread=False)
        self.create_service(Perceive, "perception/perceive", self.on_perceive, callback_group=cb)
        self.get_logger().info("perception ready")

    def on_names(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        names = {str(name): str(category) for name, category in data.items()}
        with self.lock:
            self.names = names
        self.get_logger().info(f"object names: {len(names)}")

    def on_task(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        trays = data.get("trays") if isinstance(data, dict) else None
        if not isinstance(trays, dict):
            return
        mapping = {str(color): str(tray_id) for tray_id, color in trays.items() if isinstance(color, str) and color}
        if not mapping:
            return
        with self.lock:
            self.tray_of_color = mapping
        self.get_logger().info("trays: " + ", ".join(f"{color}={tray_id}" for color, tray_id in mapping.items()))

    def store(self, cam, kind, msg):
        with self.lock:
            self.frames[cam][kind] = msg

    def snapshot(self, cam):
        with self.lock:
            f = dict(self.frames[cam])
        if not all(k in f for k in ("color", "depth", "info")):
            return None
        return f

    def cloud_in_pelvis(self, f, stride=1):
        """Organised point cloud of the depth frame in pelvis: (HxWx3 array with NaN where no depth, 4x4 cam->pelvis)."""
        depth = f["depth"]
        tf = self.tf.lookup_transform(BASE, depth.header.frame_id, Time.from_msg(depth.header.stamp),
                                      # tf at the frame time: the head may move. Just after the sim starts
                                      # /tf ran 1 s behind the depth frame (2026-11-07), and 0.5 s gave up.
                                      timeout=Duration(seconds=2.0))
        m = geo.transform_matrix(tf)
        z = geo.depth_to_m(depth)[::stride, ::stride]
        v, u = np.mgrid[0:depth.height:stride, 0:depth.width:stride]
        pts = geo.apply(m, geo.deproject(u, v, z, f["info"]).reshape(-1, 3)).reshape(z.shape + (3,))
        pts[z == 0] = np.nan
        return pts, m

    def on_perceive(self, req, res):
        cam = req.camera or "head"
        f = self.snapshot(cam) if cam in CAMERAS else None
        res.scene = SceneObjects()
        res.scene.header.frame_id = BASE
        if f is None:
            res.success, res.message = False, f"no frames from camera '{cam}'"
            return res
        res.scene.header.stamp = f["depth"].header.stamp
        try:
            self.detect(f, req.region or "table", res.scene)
            res.success, res.message = True, "ok"
        except NotImplementedError as e:
            res.success, res.message = False, f"not_implemented: {e}"
        except Exception as e:  # never kill the service: the brain decides what to do
            self.get_logger().error(f"perceive failed: {e!r}")
            res.success, res.message = False, repr(e)
        return res

    def arm_segments(self, stamp):
        """Forearms and hands in pelvis at the frame time, as pairs of points. A missing transform leaves that arm out."""
        segments = []
        for frames in ARM_FRAMES:
            try:
                poses = [geo.transform_matrix(self.tf.lookup_transform(BASE, name, Time.from_msg(stamp),
                                                                       timeout=Duration(seconds=0.2)))
                         for name in frames]
            except TransformException as e:
                self.get_logger().warning(f"no arm mask: {e}")
                continue
            points = [m[:3, 3] for m in poses] + [poses[-1][:3, 3] - FINGERS * poses[-1][:3, 2]]
            segments += list(zip(points[:-1], points[1:]))
        return segments

    def detect(self, f, region, scene):
        """Fill scene from one frame set: object categories, and trays by colour. Hood racks are M4."""
        if region != "table":
            raise NotImplementedError(f"region '{region}'")
        cloud, _ = self.cloud_in_pelvis(f)
        result = tc.detect(cloud, self.arm_segments(f["depth"].header.stamp))
        if result is None:
            raise RuntimeError("table plane not found")
        scene.table_z = result.table_z
        rgb = geo.rgb_to_np(f["color"])
        depth = f["depth"]
        if rgb.shape[0] != depth.height or rgb.shape[1] != depth.width:
            self.get_logger().warning(
                f"colour {rgb.shape[1]}x{rgb.shape[0]} differs from depth {depth.width}x{depth.height}")
            rgb = None
        with self.lock:
            self.tracker.update(result.objects)
            names, tray_of_color = dict(self.names), dict(self.tray_of_color)
        for c in result.objects:
            name, category, confidence = classify.classify_object(c.size, rgb, c.pixels, names)
            scene.objects.append(DetectedObject(
                track_id=c.track_id, name=name, category=category, confidence=float(confidence),
                yaw=float(c.yaw), top_z=float(c.top_z), partially_visible=c.partially_visible,
                center=Point(x=float(c.center[0]), y=float(c.center[1]), z=float(c.center[2])),
                size=Vector3(x=float(c.size[0]), y=float(c.size[1]), z=float(c.size[2]))))
        for tray in classify.find_trays(result.rejected, rgb, tray_of_color):
            x, y, z = tray["center"]
            yaw = tray["yaw"]
            scene.containers.append(Container(
                id=tray["id"], kind="tray", color=tray["color"],
                pose=Pose(position=Point(x=x, y=y, z=z),
                          orientation=Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))),
                size=Vector3(x=tray["size"][0], y=tray["size"][1], z=tray["size"][2])))
        named = sum(o.category != "" for o in scene.objects)
        trays = ", ".join(f"{c.id}:{c.color}" for c in scene.containers) or "none"
        if not names:
            self.get_logger().warning("no /sim/objects/names yet, categories left empty")
        self.get_logger().info(f"{named}/{len(scene.objects)} named, trays {trays}")


def main():
    rclpy.init()
    node = PerceptionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
