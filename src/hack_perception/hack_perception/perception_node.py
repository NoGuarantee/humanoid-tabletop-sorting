"""Perception service /perception/perceive. Owner: perception.

Keeps the latest colour, depth and camera_info of both cameras; on request turns them into SceneObjects in `pelvis`.
Runs on request only (not a stream): saves CPU, which counts in tie-breaks.

Status: M1 is done, the geometry is in table_clusters.py. Names, categories and containers stay empty until M2:
  M1: table plane (RANSAC) -> points above it -> clusters -> centre / yaw (PCA) / size / top_z
  M2: trays by colour (red / blue / green), category per cluster (CLIP zero-shot as a baseline)
  M3: YOLO-seg trained on sim frames of all 6 pools
  M4: racks, slots, tube caps and sizes
"""
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from hack_interfaces.msg import DetectedObject, SceneObjects
from hack_interfaces.srv import Perceive
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from . import geometry as geo
from . import table_clusters as tc

CAMERAS = {"head": "cam_head", "torso": "cam_torso"}
BASE = "pelvis"
ARM_FRAMES = (("elbow_pitch_l_link", "wrist_pitch_l_link", "left_tcp_link"),
              ("elbow_pitch_r_link", "wrist_pitch_r_link", "right_tcp_link"))
FINGERS = 0.21  # the finger tips are this far along -z of the TCP (see grasp_planner.top_down)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("hack_perception")
        self.lock = threading.Lock()
        self.frames = {cam: {} for cam in CAMERAS}
        self.tracker = tc.Tracker()
        cb = ReentrantCallbackGroup()
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
                                      timeout=Duration(seconds=0.5))  # tf at the frame time: the head may move
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
        """Fill scene.table_z and scene.objects from one frame set. Containers and hood: not yet (milestones M2, M4)."""
        if region != "table":
            raise NotImplementedError(f"region '{region}'")
        cloud, _ = self.cloud_in_pelvis(f)
        result = tc.detect(cloud, self.arm_segments(f["depth"].header.stamp))
        if result is None:
            raise RuntimeError("table plane not found")
        scene.table_z = result.table_z
        with self.lock:
            self.tracker.update(result.objects)
        for c in result.objects:
            scene.objects.append(DetectedObject(
                track_id=c.track_id, yaw=float(c.yaw), top_z=float(c.top_z), partially_visible=c.partially_visible,
                center=Point(x=float(c.center[0]), y=float(c.center[1]), z=float(c.center[2])),
                size=Vector3(x=float(c.size[0]), y=float(c.size[1]), z=float(c.size[2]))))


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
