"""Schedule worker inference from shared ROS camera frames without waiting in services."""
import json
from contextlib import contextmanager
from time import monotonic, perf_counter, time as wall_time

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import TransformException

from .geometry import transform_matrix
from .rgbd_frames import CAMERAS, PairBuffer, decode_pair, decode_view, stamp_ns
from .rgbd_response import response
from .rgbd_worker import PoseWorker, is_fresh
from .rgbd_robot_filter import JointStateBuffer

INPUT_AGE_NS = 350_000_000
SCENE_AGE_NS = 1_000_000_000
POSE_INPUT_AGE_NS = 650_000_000
POSE_AGE_NS = 1_750_000_000
SCENE_PERIOD_S = 1. / 3.
WORKER_START_LIMIT_S = 25.
WORKER_JOB_LIMIT_S = 20.


class ModelPoseService:
    def __init__(self, node, tf, model_dir, pool_override, model_poses, torch_threads, *, lock, clear_baseline,
                 on_demand=False, device='cpu', robot_self_filter=False):
        if pool_override not in (0, 4, 6):
            raise ValueError("rgbd_pool must be 0 (from task), 4 or 6")
        if not 1 <= torch_threads <= 4:
            raise ValueError("rgbd_torch_threads must be between 1 and 4")
        self.node, self.tf = node, tf
        self.pool_override = pool_override
        self.task = None
        self.generation = 0
        self.buffer = node.camera_buffer
        self.lock = lock
        self.clear_baseline = clear_baseline
        self.on_demand = on_demand
        self.robot_self_filter=bool(robot_self_filter)
        self.robot_joints=JointStateBuffer() if self.robot_self_filter else None
        self.worker_config = model_dir, model_poses, torch_threads, device, self.robot_self_filter
        self.worker = PoseWorker(*self.worker_config)
        self.worker_restarts = 0
        self.state = "worker_loading"
        self.worker_deadline = monotonic() + WORKER_START_LIMIT_S
        self.pending = None
        self.scene = None
        self.prepared_key = None
        self.last_scene_key = None
        self.last_pose_key = None
        self.next_scene_after = 0.
        self.pose = None
        self.pose_consumed = False
        self.minimum_pose_stamp_ns = 0
        self.want_pose = False
        self._init_diagnostics()
        self.pump_group = MutuallyExclusiveCallbackGroup()
        self.calls = MutuallyExclusiveCallbackGroup()
        if self.robot_self_filter:
            from sensor_msgs.msg import JointState
            self.joint_group=MutuallyExclusiveCallbackGroup()
            self.joint_subscription=node.create_subscription(JointState,'/joint_states',self.store_joint_state,50,
                                                             callback_group=self.joint_group)
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

    def _init_diagnostics(self):
        self._diag_call_seq = 0
        self._diag_pump_stage = 'not_started'
        self._diag_pump_started = self._diag_pump_completed = None
        self._diag_last_pump_timing = {}
        self._diag_last_pump_emit = -float('inf')
        self._diag_frame_received = self._diag_frame_stamp = self._diag_pair_stamp = None
        self._diag_pending_started = None
        self._diag_last_rpc_log_s = 0.
        self._diag_last_pump_log_s = 0.
        self._worker_notices = []

    @contextmanager
    def _timed_lock(self, timing, label):
        start = perf_counter()
        with self.lock:
            acquired = perf_counter()
            timing[label+'_wait_s'] = acquired-start
            try:
                yield
            finally:
                timing[label+'_held_s'] = perf_counter()-acquired

    def _diagnostic_state(self, wall, now):
        """Best-effort scalar snapshot: no extra locks or image serialization."""
        pending = self.pending
        age = lambda value: None if value is None else max(0., wall-value)
        frame_age = lambda value: None if value is None else (now-value)*1e-9
        return dict(worker_state=self.state, want_pose=self.want_pose,
            generation=self.generation,
            pending_kind=pending[0] if pending is not None else None,
            pending_stamp_ns=pending[1][1] if pending is not None else None,
            pending_job_age_s=age(self._diag_pending_started) if pending is not None else None,
            pump_stage=self._diag_pump_stage, last_pump_age_s=age(self._diag_pump_completed),
            pump_active_age_s=(age(self._diag_pump_started) if self._diag_pump_stage != 'idle' else None),
            last_pump_wall_s=dict(self._diag_last_pump_timing),
            previous_pump_log_emit_s=self._diag_last_pump_log_s,
            last_frame_received_age_s=age(self._diag_frame_received),
            latest_frame_age_sim_s=frame_age(self._diag_frame_stamp),
            latest_pair_age_sim_s=frame_age(self._diag_pair_stamp))

    def _emit_diagnostic(self, event):
        started = perf_counter()
        try:
            event.update(self._diagnostic_state(monotonic(), self.now_ns()))
            self.node.diagnostics.submit(event)
        except Exception:
            pass  # Diagnostics must not change service/worker failure behavior.
        return perf_counter()-started

    def reset(self):
        self.buffer.clear()
        self.clear_baseline()
        self.generation += 1
        self.scene = self.pose = self.prepared_key = None
        self.pose_consumed = False
        self.minimum_pose_stamp_ns = 0
        self.want_pose = False
        if getattr(self,'robot_joints',None) is not None:self.robot_joints.clear()
        self._diag_frame_received = self._diag_frame_stamp = self._diag_pair_stamp = None

    def reset_on_clock_jump(self, _jump):
        with self.lock:
            self.reset()
            self.tf.clear()

    def restart_worker(self, reason, wall):
        # Process shutdown/join and spawn can take seconds. Never hold the
        # camera/Perceive input lock across these operations.
        with self.lock:
            old_worker = self.worker
            self.state = 'worker_loading'
            self.reset()
            self.pending = self.last_scene_key = self.last_pose_key = None
            self._diag_pending_started = None
            self.worker_deadline = None
            exhausted = self.worker_restarts >= 2
            if exhausted:
                self.state = 'worker_unavailable'
            else:
                self.worker_restarts += 1
        old_worker.close()
        if exhausted:
            self.node.get_logger().error(f'RGB-D worker restart limit reached: {reason}')
            return
        try:
            new_worker = PoseWorker(*self.worker_config)
        except Exception as exc:
            with self.lock:
                self.state = 'worker_unavailable'
            self.node.get_logger().error(f'RGB-D worker restart failed: {exc!r}')
            return
        with self.lock:
            self.worker = new_worker
            self.worker_deadline = monotonic() + WORKER_START_LIMIT_S
        self.node.get_logger().warning(f'Restarting RGB-D worker ({self.worker_restarts}/2): {reason}')

    def set_task(self, data):
        with self.lock:
            if data != self.task:
                self.reset()
                self.task = data

    def store(self, camera, kind, msg, *, received_wall=None):
        # Record arrival before waiting for the shared state lock. These fields
        # are diagnostics only and never participate in exposure acceptance.
        self._diag_frame_received = monotonic() if received_wall is None else received_wall
        self._diag_frame_stamp = stamp_ns(msg.header)
        with self.lock:
            self.buffer.add(camera, kind, msg)

    def now_ns(self):
        return self.node.get_clock().now().nanoseconds

    def store_joint_state(self,msg):
        with self.lock:
            self.robot_joints.add(stamp_ns(msg.header),list(msg.name),list(msg.position))

    def bind_robot_state(self,sensors,views,key):
        with self.lock:
            if key[0]!=self.generation:raise ValueError('robot_filter_epoch_changed')
            witness=self.robot_joints.at(key[1])
            witness['generation']=key[0]
        for camera,sensor,view in zip(CAMERAS,sensors,views):
            if sensor is not None:
                sensor.update(camera=camera,stamp_ns=key[1],robot_joint_state=dict(witness),
                              optical_frame_id=view['depth'].header.frame_id)
        return sensors

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
        tf = self.tf.lookup_transform("pelvis", header.frame_id, Time.from_msg(header.stamp),
                                      timeout=Duration(seconds=.03))
        return transform_matrix(tf)

    def complete(self, item, now):
        kind, key, data, elapsed = item
        if kind == "ready":
            self.state = "ready"
            self.worker_deadline = None
            self._worker_notices.append(('info', "RGB-D worker ready"))
            return
        if kind == "failed":
            self.state = "worker_unavailable"
            self._worker_notices.append(('error', f"RGB-D worker initialization failed: {data}"))
            return
        pending, self.pending = self.pending, None
        self._diag_pending_started = None
        self.worker_deadline = None
        if kind == "job_failed":
            self._worker_notices.append(('warning', f"RGB-D worker job failed: {data}"))
            return
        if pending is None or pending[:2] != (kind, key):
            return
        self._worker_notices.append(('debug', f"RGB-D {kind} worker time {elapsed:.3f}s at stamp {key[1]}"))
        if self.on_demand or kind == 'pose':
            self._worker_notices.append(('info', json.dumps(dict(event='rgbd_worker_timing', kind=kind,
                                                        wall_s=elapsed, stamp_ns=key[1]))))
        if kind == "scene":
            self.prepared_key = key
            if is_fresh(key, self.generation, now, SCENE_AGE_NS):
                self.scene = (key, pending[2], data)
        elif kind == "pose" and is_fresh(key, self.generation, now, POSE_AGE_NS):
            self.pose = (key, data)
            self.pose_consumed = False

    def pump(self):
        started = perf_counter()
        self._diag_pump_started = monotonic()
        self._diag_pump_stage = 'clock_read'
        timing = {}
        try:
            self._pump(timing)
        finally:
            notices, self._worker_notices = self._worker_notices, []
            for level, message in notices:
                try:
                    getattr(self.node.get_logger(), level)(message)
                except Exception:
                    pass
            timing['total_s'] = perf_counter()-started
            timing['last_stage'] = self._diag_pump_stage
            self._diag_last_pump_timing = dict(timing)
            self._diag_pump_completed = monotonic()
            self._diag_pump_stage = 'idle'
            wall = self._diag_pump_completed
            since_log = wall-self._diag_last_pump_emit
            waiting = self.on_demand and (self.want_pose or self.pending is not None)
            if ((timing['total_s'] >= .25 and since_log >= 1.) or
                    (waiting and since_log >= 5.)):
                self._diag_last_pump_emit = wall
                self._diag_last_pump_log_s = self._emit_diagnostic(
                    dict(event='rgbd_pump_timing', wall_s=wall_time()))

    def _pump(self, timing):
        started = perf_counter()
        now, wall = self.now_ns(), monotonic()
        timing['clock_read_s'] = perf_counter()-started
        self._diag_pump_stage = 'worker_poll'
        started = perf_counter()
        item = self.worker.poll()
        timing['worker_poll_s'] = perf_counter()-started
        self._diag_pump_stage = 'state_lock'
        with self._timed_lock(timing, 'state_lock'):
            if item is not None:
                self._diag_pump_stage = 'complete'
                started = perf_counter()
                self.complete(item, now)
                timing['complete_s'] = perf_counter()-started
            if self.state == "worker_unavailable":
                self._diag_pump_stage = 'worker_unavailable'
                return
        if not self.worker.process.is_alive():
            self._diag_pump_stage = 'restart_exited_worker'
            self.restart_worker('process exited', wall)
            return
        if self.worker_deadline is not None and wall >= self.worker_deadline:
            self._diag_pump_stage = 'restart_worker_deadline'
            self.restart_worker('deadline exceeded', wall)
            return
        with self._timed_lock(timing, 'schedule_lock'):
            if self.state != "ready" or self.pending is not None:
                self._diag_pump_stage = 'waiting_worker'
                return
            if self.on_demand and not self.want_pose:
                self._diag_pump_stage = 'on_demand_idle'
                return

            # 6D is opt-in and keeps the exact segmentation prepared for its exposure.
            if self.want_pose and self.scene is not None:
                key, _, _ = self.scene
                if (key == self.prepared_key and key != self.last_pose_key
                        and (not self.on_demand or key[1] > self.minimum_pose_stamp_ns)
                        and is_fresh(key, self.generation, now, POSE_INPUT_AGE_NS)):
                    names = self.node.get_parameter("rgbd_model_names").value
                    request = dict(pool=self.pool(), names=[name.strip() for name in names.split(",") if name.strip()]) if names else self.pool()
                    self._diag_pump_stage = 'pose_submit'
                    started = perf_counter()
                    if self.worker.submit("pose", key, request):
                        self.pending = ("pose", key, None)
                        self._worker_notices.append(('info', json.dumps(dict(
                            event='rgbd_pose_submitted', stamp_ns=key[1], wall_s=wall_time()))))
                        self._diag_pending_started = wall
                        self.last_pose_key = key
                        self.worker_deadline = wall + WORKER_JOB_LIMIT_S
                        self.want_pose = False
                    timing['submit_s'] = perf_counter()-started
                    return
            if wall < self.next_scene_after:
                self._diag_pump_stage = 'scene_period'
                return
            self._diag_pump_stage = 'pair_select'
            started = perf_counter()
            pair = self.buffer.latest(now, INPUT_AGE_NS)
            if pair is None:
                # Prefer a synchronized pair when both cameras have one.
                pair = self.buffer.latest(now, INPUT_AGE_NS, paired=False)
            timing['pair_select_s'] = perf_counter()-started
            if pair is None:
                self._diag_pump_stage = 'no_fresh_pair'
                return
            stamp, views = pair
            self._diag_pair_stamp = stamp
            if self.on_demand and stamp <= self.minimum_pose_stamp_ns:
                self._diag_pump_stage = 'pair_before_request'
                return
            key = self.generation, stamp
            if self.last_scene_key is not None and key <= self.last_scene_key:
                self._diag_pump_stage = 'pair_already_submitted'
                return
            mapping = self.tray_mapping()

        self._diag_pump_stage = 'decode_tf'
        started = perf_counter()
        try:
            if all(view is not None for view in views):
                sensors = decode_pair(views, self.transform_at)
            else:
                sensors = [None if view is None else
                           dict(decode_view(view), T=self.transform_at(view["depth"].header)) for view in views]
            if getattr(self,'robot_self_filter',False):
                sensors=self.bind_robot_state(sensors,views,key)
        except (TransformException, ValueError) as exc:
            # The matching TF or next complete exposure may arrive before the next pump.
            self._diag_pump_stage = 'decode_tf_unavailable'
            timing['decode_tf_error'] = str(exc)
            return
        finally:
            timing['decode_tf_s'] = perf_counter()-started

        self._diag_pump_stage = 'submit_lock'
        decoded_now = self.now_ns()
        with self._timed_lock(timing, 'submit_lock'):
            self._diag_pump_stage = 'scene_submit'
            started = perf_counter()
            if (is_fresh(key, self.generation, decoded_now, INPUT_AGE_NS)
                    and self.worker.submit("scene", key, (sensors, mapping))):
                self.pending = ("scene", key, views)
                self._diag_pending_started = wall
                self.worker_deadline = wall + WORKER_JOB_LIMIT_S
                self.last_scene_key = key
                self.next_scene_after = wall + SCENE_PERIOD_S
            timing['submit_s'] = perf_counter()-started

    def scene_for(self, camera, *, now_ns=None):
        """Return the camera frame and scene measured at that exposure, if still fresh."""
        now = self.now_ns() if now_ns is None else now_ns
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

    def estimate(self, timing=None):
        timing = {} if timing is None else timing
        now = self.now_ns()
        with self._timed_lock(timing, 'estimate_lock'):
            pool = self.pool()
            if self.state != "ready":
                return response(self.state, pool=pool)
            if pool is None:
                return response("unsupported_or_missing_pool")
            if (self.pose is not None and is_fresh(self.pose[0], self.generation, now, POSE_AGE_NS)
                    and (not self.on_demand or not self.pose_consumed)):
                self.want_pose = False
                self.pose_consumed = self.on_demand
                return self.pose[1]
            # Polling an in-flight request must not queue another expensive fit
            # after the caller has already received its fresh pose.
            if self.pending is None or self.pending[0] != "pose":
                if self.on_demand and not self.want_pose:
                    self.minimum_pose_stamp_ns = now
                    self.next_scene_after = 0.
                self.want_pose = True
            return response("awaiting_fresh_pose", pool=pool)

    def on_estimate(self, _request, result):
        self._diag_call_seq += 1
        sequence = self._diag_call_seq
        arrival_wall, started = wall_time(), perf_counter()
        timing, status = {}, 'exception'
        try:
            stage = perf_counter()
            payload = self.estimate(timing)
            timing['estimate_s'] = perf_counter()-stage
            status = payload['status']
            stage = perf_counter()
            result.message = json.dumps(payload, allow_nan=False)
            result.success = status == 'ok'
            timing['json_s'] = perf_counter()-stage
            return result
        finally:
            timing['callback_s'] = perf_counter()-started
            if self.on_demand or timing['callback_s'] >= .25:
                event = dict(event='rgbd_rpc_timing', sequence=sequence,
                    arrival_wall_s=arrival_wall, arrival_perf_s=started,
                    response_ready_wall_s=wall_time(), response_ready_perf_s=perf_counter(),
                    status=status, callback_wall_s=timing,
                    previous_log_emit_s=self._diag_last_rpc_log_s)
                self._diag_last_rpc_log_s = self._emit_diagnostic(event)
