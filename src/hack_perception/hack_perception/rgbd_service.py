"""Optional ROS adapter for the sensor-only model-pose backend."""
import json
from threading import Lock

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from tf2_ros import TransformException

from .geometry import transform_matrix
from .rgbd.pose_backend import PoseBackend
from .rgbd_frames import CAMERAS, STREAMS, PairBuffer, decode_pair
from .rgbd_response import infer_response, response

MAX_AGE_NS = 1_000_000_000


class ModelPoseService:
    def __init__(self, node, tf, model_dir, pool_override):
        self.node = node
        self.tf = tf
        if pool_override not in (0, 4, 6):
            raise ValueError("rgbd_pool must be 0 (from task), 4 or 6")
        self.backend = PoseBackend(model_dir)
        self.pool_override = pool_override
        self.task_pool = None
        self.task = None
        self.generation = 0
        self.buffer = PairBuffer()
        self.lock = Lock()
        self.inputs = ReentrantCallbackGroup()
        self.calls = MutuallyExclusiveCallbackGroup()
        threshold = JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True)
        self.clock_jump = node.get_clock().create_jump_callback(threshold, post_callback=self.reset_on_clock_jump)
        for camera in CAMERAS:
            for kind, stream in STREAMS.items():
                msg_type = CameraInfo if kind.endswith("_info") else Image
                node.create_subscription(msg_type, f"/{camera}/{stream}",
                                         lambda msg, c=camera, k=kind: self.store(c, k, msg),
                                         qos_profile_sensor_data, callback_group=self.inputs)
        node.create_service(Trigger, "perception/estimate_model_poses", self.on_estimate, callback_group=self.calls)

    def reset_on_clock_jump(self, _jump):
        with self.lock:
            self.buffer.clear()
            self.generation += 1
            self.tf.clear()

    def set_task(self, data):
        with self.lock:
            if data != self.task:
                self.buffer.clear()
                self.generation += 1
                self.task = data
            source = data.get("objects_from") if isinstance(data, dict) else None
            self.task_pool = {"pool 4": 4, "pool 6": 6}.get(source) if isinstance(source, str) else None

    def store(self, camera, kind, msg):
        with self.lock:
            self.buffer.add(camera, kind, msg)

    def transform_at(self, header):
        tf = self.tf.lookup_transform("pelvis", header.frame_id, Time.from_msg(header.stamp),
                                      timeout=Duration(seconds=.25))
        return transform_matrix(tf)

    def estimate(self):
        now_ns = self.node.get_clock().now().nanoseconds
        with self.lock:
            generation = self.generation
            pool = self.pool_override or self.task_pool
            pair = self.buffer.latest(now_ns, MAX_AGE_NS)
        if pool is None:
            return response("unsupported_or_missing_pool")
        if pair is None:
            return response("no_fresh_registered_pair", pool=pool)
        stamp, frames = pair
        try:
            sensors = decode_pair(frames, self.transform_at)
        except TransformException as exc:
            self.node.get_logger().warning(f"model pose camera TF unavailable: {exc}")
            return response("missing_exposure_tf", stamp=stamp, pool=pool)
        except ValueError as exc:
            self.node.get_logger().warning(f"invalid model pose camera frame: {exc}")
            return response("invalid_rgbd", stamp=stamp, pool=pool)
        payload = infer_response(self.backend, sensors, stamp, pool)
        with self.lock:
            if self.generation != generation:
                return response("exposure_invalidated", stamp=stamp, pool=pool)
        return payload

    def on_estimate(self, _request, result):
        try:
            payload = self.estimate()
            result.message = json.dumps(payload, allow_nan=False)
            result.success = payload["status"] == "ok"
        except Exception as exc:
            self.node.get_logger().error(f"model pose inference failed: {exc!r}")
            result.message = json.dumps(response("inference_failed"))
            result.success = False
        return result
