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
import os
import threading
from time import monotonic, perf_counter, sleep

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from hack_interfaces.msg import Container, DetectedObject, SceneObjects
from hack_interfaces.srv import Perceive
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
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
from .diagnostics import DiagnosticSink
from .tray_rims import color_openings
from .cdr_image import ros_image
from .rgbd.geometry import table_z_at
from .rgbd_frames import STREAMS, PairBuffer, decode_image, stamp_ns

CAMERAS = {"head": "cam_head", "torso": "cam_torso"}
BASE = "pelvis"
ARM_FRAMES = (("elbow_pitch_l_link", "wrist_pitch_l_link", "left_tcp_link"),
              ("elbow_pitch_r_link", "wrist_pitch_r_link", "right_tcp_link"))
FINGERS = 0.21  # along -z of the TCP; keep equal to grasp_planner.FINGERTIP
NAMES_WAIT_S = 3.0  # wall s: /sim/objects/names is latched and normally arrives within a second of start
STILL_WINDOW = 0.5  # s: a frame at rest this long may take the latest transform (lookup_at)
STILL_TOLERANCE = 0.002  # max change of any 4x4 entry over that window: 2 mm, about 0.1°


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("hack_perception")
        self.lock = threading.RLock()
        self.perceive_calls = 0  # Perceive callbacks running now, in the timing record
        self.frame_arrived = threading.Condition(self.lock)
        self.diagnostics = DiagnosticSink(self.get_logger())
        self.declare_parameter("perceive_frame_wait_s", .75)
        raw_images = self.declare_parameter('serialized_images', True, ParameterDescriptor(read_only=True)).value
        self.clear_baseline()
        self.names = {}  # object name -> category, from /sim/objects/names
        self.names_ready = threading.Event()
        self.tray_of_color = dict(classify.DEFAULT_TRAY)  # replaced by /sim/task when it arrives
        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/sim/objects/names", self.on_names, latched, callback_group=cb)
        self.create_subscription(String, "/sim/task", self.on_task, latched, callback_group=cb)
        self.tf = Buffer()
        # /tf on its own node and thread. On this node's executor every /tf message became a task in the shared
        # thread pool (tf2_ros puts it in a reentrant group): /tf lagged 1.2 s, and under load /clock and the camera
        # callbacks waited behind it for over 10 s (eval 2026-11-08).
        self.tf_listener = TransformListener(self.tf, None, spin_thread=True)
        self.create_service(Perceive, "perception/perceive", self.on_perceive, callback_group=cb)
        self.model_poses = None
        model_dir = self.declare_parameter("rgbd_model_dir", os.environ.get("NSSR_RGBD_MODEL_DIR", "")).value
        pool = self.declare_parameter("rgbd_pool", 0).value
        enable_poses = self.declare_parameter("rgbd_enable_model_poses", False).value
        on_demand = self.declare_parameter("rgbd_on_demand", False).value
        device = self.declare_parameter("rgbd_device", os.environ.get('NSSR_RGBD_DEVICE', 'cpu')).value
        self.declare_parameter("rgbd_model_names", "", ParameterDescriptor(read_only=True))
        torch_threads = self.declare_parameter("rgbd_torch_threads", 2).value
        if model_dir:
            try:
                from .rgbd_service import ModelPoseService

                self.model_poses = ModelPoseService(self, self.tf, model_dir, pool, enable_poses, torch_threads,
                                                    lock=self.lock, clear_baseline=self.clear_baseline,
                                                    on_demand=on_demand, device=device)
            except Exception as e:
                self.get_logger().error(f"could not enable RGB-D model pose service: {e!r}")
        camera_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        camera_group = MutuallyExclusiveCallbackGroup()
        for cam, topic in CAMERAS.items():
            for kind, stream in STREAMS.items():
                msg_type = CameraInfo if kind.endswith("_info") else Image
                self.create_subscription(msg_type, f"/{topic}/{stream}",
                                         lambda m, c=cam, k=kind, raw=raw_images and msg_type is Image:
                                             self.store_image(c, k, m, raw),
                                         camera_qos, callback_group=camera_group,
                                         raw=raw_images and msg_type is Image)
        # Motion builds the cuRobo world from this, between its own plans. Latched: a plan after the scan still sees it.
        self.scene_pub = self.create_publisher(SceneObjects, "perception/scene", latched)
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
        self.names_ready.set()
        self.get_logger().info(f"object names: {len(names)}")

    def on_task(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if self.model_poses is not None:
            self.model_poses.set_task(data)
        trays = data.get("trays") if isinstance(data, dict) else None
        if not isinstance(trays, dict):
            return
        mapping = {str(color): str(tray_id) for tray_id, color in trays.items() if isinstance(color, str) and color}
        if not mapping:
            return
        with self.lock:
            self.tray_of_color = mapping
        self.get_logger().info("trays: " + ", ".join(f"{color}={tray_id}" for color, tray_id in mapping.items()))

    def clear_baseline(self):
        # Callers hold the shared input lock once the node is running.
        self.frames = {cam: {} for cam in CAMERAS}
        self.tracker = tc.Tracker()
        if not hasattr(self, 'camera_buffer'):
            self.camera_buffer = PairBuffer()
        else:
            self.camera_buffer.clear()

    def store(self, cam, kind, msg):
        received_wall = monotonic()
        with self.lock:
            if kind != "depth_info":
                key = "info" if kind == "color_info" else kind
                self.frames[cam][key] = msg
            if self.model_poses is not None:
                self.model_poses.store(CAMERAS[cam], kind, msg, received_wall=received_wall)
            else:
                self.camera_buffer.add(CAMERAS[cam], kind, msg)
            self.frame_arrived.notify_all()

    def store_image(self, cam, kind, msg, raw):
        if raw:
            try:
                msg = ros_image(msg)
            except (ValueError, UnicodeError) as exc:
                self.diagnostics.submit(dict(event='image_decode_error', camera=cam, reason=str(exc)))
                return
        self.store(cam, kind, msg)

    def snapshot(self, cam, *, now_ns=None, after_ns=0, max_age_ns=1_000_000_000):
        now = self.get_clock().now().nanoseconds if now_ns is None else now_ns
        with self.lock:
            view = self.camera_buffer.latest_for(CAMERAS[cam], now, max_age_ns, after_ns=after_ns)
            epoch = self.model_poses.generation if self.model_poses is not None else None
        if view is None:
            return None, None
        if len({(stamp_ns(m.header), m.header.frame_id) for m in view.values()}) != 1:
            return None, None
        f = dict(color=view['color'], depth=view['depth'], info=view['color_info'])
        return f, (epoch, stamp_ns(f["depth"].header))

    def lookup_at(self, frame, stamp, sim_wait, wall_cap=None):
        """Transform frame -> pelvis at stamp, waiting up to sim_wait seconds of /clock for /tf to catch up.

        tf2's own timeout counts wall time, and the sim runs below real time. Even waiting on /clock, this node's
        buffer once stayed 1.07 s behind the depth frame while /tf on the wire kept within 30 ms of it
        (2026-11-07, tools/stamp_probe.py), and 1.1-1.2 s mid-task in eval (2026-11-08). Then the latest
        transform stands in, if the frame did not move over STILL_WINDOW before it: the brain looks with the
        head and waist at rest. wall_cap bounds the wait if /clock stalls: 30 s once held a Perceive call past the
        brain's 6 s and the node past its shutdown (eval 2026-11-08), so it is a few times sim_wait."""
        if wall_cap is None:
            wall_cap = max(1.0, 3.0 * sim_wait)
        t = Time.from_msg(stamp)
        start, wall = self.get_clock().now(), monotonic()
        while not self.tf.can_transform(BASE, frame, t):
            if (self.get_clock().now() - start).nanoseconds > sim_wait * 1e9:
                break
            if monotonic() - wall > wall_cap:
                self.get_logger().warning(f"{frame}: /clock here moved "
                    f"{(self.get_clock().now() - start).nanoseconds * 1e-9:.2f} s in {wall_cap:.1f} s wall")
                break
            sleep(0.02)
        try:
            return self.tf.lookup_transform(BASE, frame, t)
        except TransformException as e:
            latest = self.tf.lookup_transform(BASE, frame, Time())
            t_latest = Time.from_msg(latest.header.stamp)
            before = self.tf.lookup_transform(BASE, frame, t_latest - Duration(seconds=STILL_WINDOW))
            moved = geo.transform_matrix(latest) - geo.transform_matrix(before)
            lag = (t - t_latest).nanoseconds * 1e-9
            now = (self.get_clock().now() - t).nanoseconds * 1e-9
            self.get_logger().warning(f"{frame}: /tf here {lag:.2f} s behind the frame, frame {now:.2f} s old, "
                                      f"moved {np.abs(moved).max():.4f} over {STILL_WINDOW} s")
            if np.abs(moved).max() > STILL_TOLERANCE:
                raise e
            return latest


    def cloud_in_pelvis(self, f, stride=1):
        """Organised point cloud of the depth frame in pelvis: (HxWx3 array with NaN where no depth, 4x4 cam->pelvis)."""
        depth = f["depth"]
        # tf at the frame time: the head may move. Just after the sim starts /tf ran 1 s behind the depth frame
        # (2026-11-07), and 0.5 s gave up.
        m = geo.transform_matrix(self.lookup_at(depth.header.frame_id, depth.header.stamp, sim_wait=2.0))
        z = geo.depth_to_m(depth)[::stride, ::stride]
        v, u = np.mgrid[0:depth.height:stride, 0:depth.width:stride]
        pts = geo.apply(m, geo.deproject(u, v, z, f["info"]).reshape(-1, 3)).reshape(z.shape + (3,))
        pts[z == 0] = np.nan
        return pts, m

    def on_perceive(self, req, res):
        started = perf_counter()
        timing = {}
        with self.lock:
            self.perceive_calls += 1
            timing['concurrent_calls'] = self.perceive_calls
        try:
            return self._perceive(req, res, timing)
        finally:
            with self.lock:
                self.perceive_calls -= 1
            elapsed = perf_counter() - started
            self.diagnostics.submit(dict(event='perceive_timing', callback_wall_s=elapsed,
                stages=timing, success=res.success, message=res.message))

    def _perceive(self, req, res, timing):
        cam = req.camera or "head"
        # Without the latched names no object gets a category, and the brain finds nothing to sort: it once
        # called Perceive 0.5 s after start, before the names arrived (eval 2026-11-08, layouts 2 and 3).
        if not self.names_ready.is_set():
            waited = monotonic()
            if not self.names_ready.wait(timeout=NAMES_WAIT_S):
                self.get_logger().warning(f"perceive: no /sim/objects/names after {NAMES_WAIT_S} s, objects unnamed")
            timing['names_wait_s'] = monotonic()-waited
        requested_ns = self.get_clock().now().nanoseconds
        timing['request_stamp_ns'] = requested_ns
        start = monotonic()
        deadline = start + max(0., min(1., self.get_parameter('perceive_frame_wait_s').value))
        f, key, model, stale = None, None, None, False
        # A service called after LOOK_TABLE must not return a pre-motion frame.
        # Condition.wait releases the input lock so camera and TF callbacks run.
        while cam in CAMERAS:
            # Never acquire the ROS clock mutex under the input lock: clock
            # reset callbacks acquire those locks in the opposite order.
            now = self.get_clock().now().nanoseconds
            with self.frame_arrived:
                if self.model_poses is not None:
                    model = self.model_poses.scene_for(CAMERAS[cam], now_ns=now)
                    if model is not None and model[0][1] >= requested_ns:
                        break
                    model = None
                f, key = self.snapshot(cam, now_ns=now, after_ns=requested_ns)
                if f is not None and key[1] >= requested_ns:
                    break
                remaining = deadline - monotonic()
                if remaining <= 0:
                    # No exposure after the request: the last complete one, of any age,
                    # beats an empty answer that stops the mission. The buffer is
                    # cleared on a clock reset, and the cloud uses TF at its own stamp.
                    f, key = self.snapshot(cam, now_ns=now, max_age_ns=2**63)
                    stale = f is not None
                    break
                # /clock can catch up without another camera callback. Retain
                # older complete exposures instead of overwriting with a future one.
                self.frame_arrived.wait(timeout=min(remaining, .005))
        timing['frame_wait_s'] = monotonic()-start
        if model is not None:
            key, f, geometry = model
        else:
            geometry = (None, [])
        res.scene = SceneObjects()
        res.scene.header.frame_id = BASE
        if f is None:
            res.success, res.message = False, f"no frames from camera '{cam}'"
            return res
        header = f["depth"].header
        if self.model_poses is not None and not stale and not self.model_poses.valid_scene(key):
            res.success, res.message = False, "no fresh RGB-D exposure"
            return res
        res.scene.header.stamp = header.stamp
        try:
            self.detect(f, req.region or "table", res.scene, geometry, key[0], timing)
            if self.model_poses is not None and not stale and not self.model_poses.valid_scene(key):
                res.scene = SceneObjects()
                res.scene.header.frame_id = BASE
                res.success, res.message = False, "RGB-D worker exposure expired"
                return res
            age = (self.get_clock().now().nanoseconds-key[1])*1e-9
            timing['response_age_sim_s'], timing['stale_frame'] = age, stale
            res.success = True
            res.message = f"stale frame: none after the request, using one {age:.2f} s old" if stale else "ok"
            if stale:
                self.get_logger().warn(f"perceive({cam}): {res.message}")
            self.scene_pub.publish(res.scene)
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
                poses = [geo.transform_matrix(self.lookup_at(name, stamp, sim_wait=0.2)) for name in frames]
            except TransformException as e:
                self.get_logger().debug(f"no arm mask: {e}")
                continue
            points = [m[:3, 3] for m in poses] + [poses[-1][:3, 3] - FINGERS * poses[-1][:3, 2]]
            segments += list(zip(points[:-1], points[1:]))
        return segments

    def detect(self, f, region, scene, geometry, epoch, timing):
        """Fill scene from one frame set: object categories, and trays by colour. Hood racks are M4."""
        if region != "table":
            raise NotImplementedError(f"region '{region}'")
        started = perf_counter()
        cloud, camera_tf = self.cloud_in_pelvis(f)
        timing['cloud_tf_s'] = perf_counter()-started
        started = perf_counter()
        arms = self.arm_segments(f["depth"].header.stamp)
        timing['arm_tf_s'] = perf_counter()-started
        started = perf_counter()
        result = tc.detect(cloud, arms)
        timing['cluster_s'] = perf_counter()-started
        started = perf_counter()
        with self.lock:
            names, tray_of_color = dict(self.names), dict(self.tray_of_color)
        plane, model_trays = geometry
        if result is None and plane is None:
            raise RuntimeError("table plane not found")
        scene.table_z = table_z_at(plane, *tc.TABLE_REF_XY) if plane is not None else result.table_z
        if plane is not None:
            normal, offset = plane
        else:
            normal, offset = result.plane.normal, -float(result.plane.normal @ result.plane.point)
        normal = np.asarray(normal, dtype=float)
        scale = float(np.linalg.norm(normal)) * (1.0 if normal[2] > 0 else -1.0)
        scene.table_plane = [*(float(v) / scale for v in normal), float(offset) / scale]  # motion's collision world
        rgb = geo.rgb_to_np(f["color"])
        depth = f["depth"]
        if rgb.shape[0] != depth.height or rgb.shape[1] != depth.width:
            self.get_logger().warning(
                f"colour {rgb.shape[1]}x{rgb.shape[0]} differs from depth {depth.width}x{depth.height}")
            rgb = None
        objects = result.objects if result is not None else []
        with self.lock:
            if self.model_poses is not None and self.model_poses.generation != epoch:
                raise RuntimeError("RGB-D exposure invalidated")
            self.tracker.update(objects)
        for c in objects:
            name, category, confidence = classify.classify_object(c.size, rgb, c.pixels, names)
            scene.objects.append(DetectedObject(
                track_id=c.track_id, name=name, category=category, confidence=float(confidence),
                yaw=float(c.yaw), top_z=float(c.top_z), partially_visible=c.partially_visible,
                center=Point(x=float(c.center[0]), y=float(c.center[1]), z=float(c.center[2])),
                size=Vector3(x=float(c.size[0]), y=float(c.size[1]), z=float(c.size[2]))))
        rejected = result.rejected if result is not None else []
        trays = classify.merge_trays(model_trays, rejected, rgb, tray_of_color)
        missing = {color: ident for color, ident in tray_of_color.items()
                   if ident not in {tray[0] for tray in trays}}
        if result is not None and missing:
            dtype = '>u2' if f['depth'].is_bigendian else '<u2'
            depth_mm = decode_image(f['depth'], '16UC1', dtype, 1)[..., 0].astype(np.uint16, copy=False)
            sensor = dict(rgb=rgb, depth=depth_mm,
                          K=np.asarray(f['info'].k).reshape(3, 3), T=camera_tf)
            measured = color_openings(sensor, cloud, result.plane, missing) if rgb is not None else []
            trays.extend((tray.id, tray.color, tray.center, tray.yaw, tray.size) for tray in measured)
        for tray_id, color, center, yaw, size in trays:
            x, y, z = map(float, center)
            scene.containers.append(Container(
                id=tray_id, kind="tray", color=color,
                pose=Pose(position=Point(x=x, y=y, z=z),
                          orientation=Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))),
                size=Vector3(x=size[0], y=size[1], z=size[2])))
        named = sum(o.category != "" for o in scene.objects)
        trays = ", ".join(f"{c.id}:{c.color}" for c in scene.containers) or "none"
        timing['classify_trays_s'] = perf_counter()-started
        timing['named_objects'] = named
        timing['trays'] = trays


def main():
    rclpy.init()
    node = PerceptionNode()
    # Perceive, the RGB-D pump, the cameras, /tf and /clock each hold a thread at times. With 4, /clock and camera
    # frames once stopped for over 10 s while a Perceive call waited on them (eval 2026-11-08).
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node.model_poses is not None:
            node.model_poses.close()
        node.diagnostics.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
