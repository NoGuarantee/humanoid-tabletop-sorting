"""Schedule worker inference from shared ROS camera frames without waiting in services."""
import json
from time import monotonic

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import TransformException

from .geometry import transform_matrix
from .rgbd_frames import CAMERAS, PairBuffer, decode_pair, decode_view
from .rgbd_response import response
from .rgbd_worker import PoseWorker, is_fresh

INPUT_AGE_NS = 350_000_000
SCENE_AGE_NS = 1_000_000_000
POSE_INPUT_AGE_NS = 650_000_000
POSE_AGE_NS = 1_750_000_000
SCENE_PERIOD_S = 1. / 3.
WORKER_START_LIMIT_S = 25.
WORKER_JOB_LIMIT_S = 8.


class ModelPoseService:
    def __init__(self, node, tf, model_dir, pool_override, model_poses, torch_threads, *, lock, clear_baseline):
        if pool_override not in (0, 4, 6):
            raise ValueError("rgbd_pool must be 0 (from task), 4 or 6")
        if not 1 <= torch_threads <= 4:
            raise ValueError("rgbd_torch_threads must be between 1 and 4")
        self.node, self.tf = node, tf
        self.pool_override = pool_override
        self.task = None
        self.generation = 0
        self.buffer = PairBuffer()
        self.lock = lock
        self.clear_baseline = clear_baseline
        self.worker = PoseWorker(model_dir, model_poses, torch_threads)
        self.state = "worker_loading"
        self.worker_deadline = monotonic() + WORKER_START_LIMIT_S
        self.pending = None
        self.scene = None
        self.prepared_key = None
        self.last_scene_key = None
        self.last_pose_key = None
        self.next_scene_after = 0.
        self.pose = None
        self.want_pose = False
        self.pump_group = MutuallyExclusiveCallbackGroup()
        self.calls = MutuallyExclusiveCallbackGroup()
        threshold = JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True)
        self.clock_jump = node.get_clock().create_jump_callback(threshold, post_callback=self.reset_on_clock_jump)
        self.timer = node.create_timer(.05, self.pump, callback_group=self.pump_group,
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))
        if model_poses:
            node.create_service(Trigger, "perception/estimate_model_poses", self.on_estimate,
                                callback_group=self.calls)

    def close(self):
        self.timer.cancel()
        self.clock_jump.unregister()
        self.worker.close()

    def reset(self):
        self.buffer.clear()
        self.clear_baseline()
        self.generation += 1
        self.scene = self.pose = self.prepared_key = None
        self.want_pose = False

    def reset_on_clock_jump(self, _jump):
        with self.lock:
            self.reset()
            self.tf.clear()

    def set_task(self, data):
        with self.lock:
            if data != self.task:
                self.reset()
                self.task = data

    def store(self, camera, kind, msg):
        with self.lock:
            self.buffer.add(camera, kind, msg)

    def now_ns(self):
        return self.node.get_clock().now().nanoseconds

    def pool(self):
        source = self.task.get("objects_from") if isinstance(self.task, dict) else None
        task_pool = {"pool 4": 4, "pool 6": 6}.get(source) if isinstance(source, str) else None
        return self.pool_override or task_pool

    def tray_mapping(self):
        trays = self.task.get("trays") if isinstance(self.task, dict) else None
        if isinstance(trays, dict):
            return {color: tray_id for tray_id, color in trays.items() if isinstance(color, str) and color}
        return None

    def transform_at(self, header):
        tf = self.tf.lookup_transform("pelvis", header.frame_id, Time.from_msg(header.stamp))
        return transform_matrix(tf)

    def complete(self, item, now):
        kind, key, data, elapsed = item
        if kind == "ready":
            self.state = "ready"
            self.worker_deadline = None
            self.node.get_logger().info("RGB-D worker ready")
            return
        if kind == "failed":
            self.state = "worker_unavailable"
            self.node.get_logger().error(f"RGB-D worker initialization failed: {data}")
            return
        pending, self.pending = self.pending, None
        self.worker_deadline = None
        if kind == "job_failed":
            self.node.get_logger().warning(f"RGB-D worker job failed: {data}")
            return
        if pending is None or pending[:2] != (kind, key):
            return
        self.node.get_logger().debug(f"RGB-D {kind} worker time {elapsed:.3f}s at stamp {key[1]}")
        if kind == "scene":
            self.prepared_key = key
            if is_fresh(key, self.generation, now, SCENE_AGE_NS):
                self.scene = (key, pending[2], data)
        elif kind == "pose" and is_fresh(key, self.generation, now, POSE_AGE_NS):
            self.pose = (key, data)

    def pump(self):
        now, wall = self.now_ns(), monotonic()
        item = self.worker.poll()
        with self.lock:
            if item is not None:
                self.complete(item, now)
            if self.state == "worker_unavailable":
                return
            if not self.worker.process.is_alive():
                self.state = "worker_unavailable"
                self.pending = self.scene = self.pose = None
                self.node.get_logger().warning("RGB-D worker exited, using baseline perception")
                return
            if self.worker_deadline is not None and wall >= self.worker_deadline:
                self.worker.process.terminate()
                self.state = "worker_unavailable"
                self.pending = self.scene = self.pose = None
                self.node.get_logger().warning("RGB-D worker deadline exceeded, using baseline perception")
                return
            if self.state != "ready" or self.pending is not None:
                return

            # 6D is opt-in and keeps the exact segmentation prepared for its exposure.
            if self.want_pose and self.scene is not None:
                key, _, _ = self.scene
                if (key == self.prepared_key and key != self.last_pose_key
                        and is_fresh(key, self.generation, now, POSE_INPUT_AGE_NS)):
                    if self.worker.submit("pose", key, self.pool()):
                        self.pending = ("pose", key, None)
                        self.last_pose_key = key
                        self.worker_deadline = wall + WORKER_JOB_LIMIT_S
                        self.want_pose = False
                    return
            if wall < self.next_scene_after:
                return
            pair = self.buffer.latest(now, INPUT_AGE_NS)
            if pair is None:
                # Prefer a synchronized pair when both cameras have one.
                pair = self.buffer.latest(now, INPUT_AGE_NS, paired=False)
            if pair is None:
                return
            stamp, views = pair
            key = self.generation, stamp
            if self.last_scene_key is not None and key <= self.last_scene_key:
                return
            mapping = self.tray_mapping()

        try:
            if all(view is not None for view in views):
                sensors = decode_pair(views, self.transform_at)
            else:
                sensors = [None if view is None else
                           dict(decode_view(view), T=self.transform_at(view["depth"].header)) for view in views]
        except (TransformException, ValueError):
            # The matching TF or next complete exposure may arrive before the next pump.
            return

        with self.lock:
            if key[0] == self.generation and self.worker.submit("scene", key, (sensors, mapping)):
                self.pending = ("scene", key, views)
                self.worker_deadline = wall + WORKER_JOB_LIMIT_S
                self.last_scene_key = key
                self.next_scene_after = wall + SCENE_PERIOD_S

    def scene_for(self, camera):
        """Return the camera frame and scene measured at that exposure, if still fresh."""
        now = self.now_ns()
        with self.lock:
            if self.scene is None or not is_fresh(self.scene[0], self.generation, now, SCENE_AGE_NS - INPUT_AGE_NS):
                return None
            key, views, geometry = self.scene
            view = views[CAMERAS.index(camera)]
            if view is None or geometry[0] is None:
                return None
            frames = dict(color=view["color"], depth=view["depth"], info=view["color_info"])
            return key, frames, geometry

    def valid_scene(self, key):
        now = self.now_ns()
        with self.lock:
            return is_fresh(key, self.generation, now, SCENE_AGE_NS)

    def estimate(self):
        now = self.now_ns()
        with self.lock:
            pool = self.pool()
            if self.state != "ready":
                return response(self.state, pool=pool)
            if pool is None:
                return response("unsupported_or_missing_pool")
            if self.pose is not None and is_fresh(self.pose[0], self.generation, now, POSE_AGE_NS):
                return self.pose[1]
            self.want_pose = True
            return response("awaiting_fresh_pose", pool=pool)

    def on_estimate(self, _request, result):
        payload = self.estimate()
        result.message = json.dumps(payload, allow_nan=False)
        result.success = payload["status"] == "ok"
        return result
