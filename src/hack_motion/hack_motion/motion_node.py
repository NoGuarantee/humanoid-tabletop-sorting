"""Motion server: actions MoveJoints, Hand, MoveTcp. Owner: motion.

MoveJoints: synchronised arrival of all joints, closed loop on /joint_states against the PD undershoot.
Hand:       named presets, reports whether the fingers stopped on an object.
MoveTcp:    cuRobo (parameter `planner`, default curobo). `direct` is one IK solution and a collision-free
            joint-space path to it, `linear` is a Cartesian trajectory. The reported error is the TF of
            the TCP, not the joint error.
"""
import math
import json
import os
import threading
import time
import traceback

import rclpy
from ament_index_python.packages import get_package_share_directory
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from bodyctrl_msgs.srv import SetForce, SetSpeed
from hack_interfaces.action import Hand, MoveJoints, MoveTcp
from hack_interfaces.msg import SceneObjects
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .curobo_planner import Unreachable
from .robot import ARM_JOINTS, HAND_PRESETS, JOINTS, MAX_SPEED, TCP_LINK, TRAY_IDS, clip, raise_steps

PERIOD = 0.05  # s, real time between control checks
# Inspire actuator 6, thumb rotation. It turns across before the other fingers close.
THUMB = 5
# /inspire_hand/set_speed: 1.0 is a full stroke in 0.8 s. Half of that keeps the close from kicking the object out.
HAND_CLOSE_SPEED = 0.5
HAND_STROKE_S = 0.8
FINGER_MOVE = 0.05  # a finger this far from where it was is actually closing or turning


def _closing(current, target):
    """A preset that pulls any finger shut. Unknown state counts too: pinch and power both close."""
    if current is None:
        return any(v < 0.9 for v in target)
    return any(target[i] < current[i] - FINGER_MOVE for i in range(6))


def _thumb_first(current, target):
    """Thumb rotation and another finger both have to move. The thumb goes first."""
    if current is None:
        return False
    turns = target[THUMB] < current[THUMB] - FINGER_MOVE
    others = any(abs(target[i] - current[i]) > FINGER_MOVE for i in range(5))
    return turns and others


# 1/s, integrates the PD undershoot. The arm actuators settle with tau = kd/kp = 0.25 s, so this is damping 0.7.
COMPENSATION_RATE = 2.0
MAX_COMPENSATION = 0.3  # rad added on top of the target at most
HOME_TOL = 0.1  # rad, every arm joint within this of 0 -> the arm hangs at home
READY_TOL = 0.05  # rad: a `ready` move that timed out this close to the raised pose still counts
HELD_TOL = 0.05  # rad, the arm MoveTcp did not ask for: it only holds its pose
# Joint step of one follow segment. 0.3 rad left the 10 cm grasp descent (21 planned waypoints, no joint
# past 0.3) as a single joint-space chord, so the hand did not travel the straight line cuRobo had planned.
PATH_STEP = 0.1


class MotionNode(Node):
    def __init__(self):
        super().__init__("hack_motion")
        self.lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._execution = threading.local()
        self.q = {}
        self.compensation = {}  # joint -> rad on top of the target, kept between moves (see move_joints)
        self.hand_state = {"left": None, "right": None}
        self.task_trays = None  # trays of the task from /sim/task, None until it arrives
        self.scene_objects = ()  # neighbours from the latest perception/scene, tuples for the cuRobo world
        self.table_plane = None  # (a, b, c, d) of the table in pelvis, the latest perception/scene that had one
        self.seen_trays = {}  # tray id -> tray tuple (robot.TRAY_SIZE), the last pose perception measured
        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10, callback_group=cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/sim/task", self.on_task, latched, callback_group=cb)
        self.create_subscription(SceneObjects, "perception/scene", self.on_scene, latched, callback_group=cb)
        for side in ("left", "right"):
            self.create_subscription(JointState, f"/inspire_hand/state/{side}_hand",
                                     lambda m, s=side: self.on_hand(s, m), 10, callback_group=cb)
        self.cmd = {g: self.create_publisher(CmdSetMotorPosition, f"/{g}/cmd_pos", 10)
                    for g in ("arm", "head", "waist")}
        self.hand_cmd = {s: self.create_publisher(JointState, f"/inspire_hand/ctrl/{s}_hand", 10)
                         for s in ("left", "right")}
        self.hand_speed = {s: self.create_client(SetSpeed, f"/inspire_hand/set_speed/{s}_hand", callback_group=cb)
                           for s in ("left", "right")}
        self.hand_force = {s: self.create_client(SetForce, f"/inspire_hand/set_force/{s}_hand", callback_group=cb)
                           for s in ("left", "right")}
        ActionServer(self, MoveJoints, "motion/move_joints", self.exec_move_joints, callback_group=cb,
                     cancel_callback=lambda _: CancelResponse.ACCEPT)
        ActionServer(self, Hand, "motion/hand", self.exec_hand, callback_group=cb,
                     cancel_callback=lambda _: CancelResponse.ACCEPT)
        ActionServer(self, MoveTcp, "motion/move_tcp", self.exec_move_tcp, callback_group=cb,
                     cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.declare_parameter("planner", "curobo")  # curobo | ik (ik is the unused fallback, see ik.py)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._planner = None
        self._tcp_lock = threading.Lock()
        if self.get_parameter("planner").value == "curobo":
            threading.Thread(target=self._preload, daemon=True).start()
        self.get_logger().info("motion server ready")

    def _preload(self):
        """Build the planner while brain is still scanning the scene, so the first MoveTcp does not wait
        for it. On failure move_tcp tries again and reports the error."""
        self._startup("preload starts")
        if not self._wait_joints(timeout=60.0):
            return
        self._startup("first /joint_states")
        with self._tcp_lock:
            try:
                with self.lock:
                    snapshot = dict(self.q)
                self.get_logger().info(self._planner_obj().load(snapshot))
                self._startup("planner loaded")
            except Exception:
                self.get_logger().error(traceback.format_exc())

    def _startup(self, what):
        """Start-up timeline for tools/startup_report.py: the log prefix has the wall clock, this adds the sim clock."""
        self.get_logger().info(f"startup: {what}, sim {self.now():.2f}")

    # ---------------------------------------------------------------- state
    def on_task(self, msg):
        """Which trays this task has (sort2 has no tray_center, objects may lie there). Their poses come from perception."""
        try:
            trays = json.loads(msg.data).get("trays", {})
        except (ValueError, AttributeError) as e:
            self.get_logger().error(f"cannot read /sim/task: {e!r}")
            trays = {}
        self.task_trays = tuple(sorted(name for name in trays if name in TRAY_IDS))
        self.get_logger().info(f"trays of the task: {list(self.task_trays)}")

    def on_scene(self, msg):
        """The latest objects, as cuboids for the next plan. The one under the goal is left out there, not here."""
        objs = tuple(
            (o.track_id or f"obj{i}", float(o.center.x), float(o.center.y), float(o.center.z), float(o.yaw),
             float(o.size.x), float(o.size.y), float(o.size.z), float(o.top_z))
            for i, o in enumerate(msg.objects))
        plane = tuple(float(v) for v in msg.table_plane)
        seen = {}
        for c in msg.containers:
            if c.kind != "tray" or c.id not in TRAY_IDS:
                continue
            q = c.pose.orientation
            yaw = 2.0 * math.atan2(q.z, q.w)  # perception writes the yaw alone, about the pelvis z
            seen[c.id] = (c.id, float(c.pose.position.x), float(c.pose.position.y), float(c.pose.position.z), yaw,
                          float(c.size.x), float(c.size.y), float(c.size.z))
        found = len(plane) == 4 and plane[2] > 0.0
        with self.lock:
            self.scene_objects = objs
            if found:
                self.table_plane = plane
            # A tray the arm hides keeps the pose it was last seen at: trays are not moved during the task.
            self.seen_trays.update(seen)
        table = f"z = {-plane[3] / plane[2]:.4f} + {-plane[0] / plane[2]:.4f} x" if found else "none"
        self.get_logger().info(f"perception scene: {len(objs)} objects, trays {sorted(seen)}, table {table}")

    def _world(self):
        """(table plane, trays) for the cuRobo world: what perception measured, limited to the trays of the task."""
        with self.lock:
            trays = [t for name, t in sorted(self.seen_trays.items())
                     if self.task_trays is None or name in self.task_trays]
            return self.table_plane, tuple(trays)

    def on_joints(self, msg):
        with self.lock:
            self.q.update(zip(msg.name, msg.position))

    def on_hand(self, side, msg):
        with self.lock:
            self.hand_state[side] = msg

    def joints(self, names):
        with self.lock:
            return [self.q.get(n) for n in names]

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def send(self, targets, speeds):
        by_group = {}
        for name, pos in targets.items():
            motor, group, _, _ = JOINTS[name]
            # spd 0 means max speed for the driver, so never send exactly 0
            by_group.setdefault(group, []).append(
                SetMotorPosition(name=motor, pos=float(pos), spd=float(max(speeds[name], 0.01)), cur=1.0))
        for group, cmds in by_group.items():
            msg = CmdSetMotorPosition()
            msg.cmds = cmds
            self.cmd[group].publish(msg)

    # ---------------------------------------------------------------- MoveJoints
    def move_joints(self, names, positions, speed=0.0, tol=0.0, max_time=0.0, feedback=None, held=()):
        """Blocking joint move. Returns (success, max_error, message). Joints in `held` only need HELD_TOL."""
        if self.cancelled():
            return False, 0.0, 'cancelled'
        if not names or not all(math.isfinite(p) for p in positions):
            return False, 0.0, 'invalid joint goal'
        unknown = [n for n in names if n not in JOINTS]
        if unknown:
            return False, 0.0, f"unknown joints {unknown}"
        if len(names) != len(positions):
            return False, 0.0, "joints and positions differ in length"
        speed = min(speed or MAX_SPEED, MAX_SPEED)
        tol = tol or 0.02
        tols = {n: max(tol, HELD_TOL) if n in held else tol for n in names}
        goal = {n: clip(n, p) for n, p in zip(names, positions)}
        start = self.joints(names)
        if any(v is None for v in start):
            return False, 0.0, "no /joint_states yet"
        dist = {n: abs(goal[n] - q0) for n, q0 in zip(names, start)}
        duration = max(max(dist.values()) / speed, 0.1)
        speeds = {n: dist[n] / duration for n in names}  # every joint arrives at the same time
        max_time = max_time or (2.0 * duration + 3.0)

        # The droop changes slowly with the pose, so a move starts from what the last one needed. Without this, every
        # segment of a path lets the arm sag again before anything pushes it back.
        cmd = {n: clip(n, goal[n] + self.compensation.get(n, 0.0)) for n in names}
        self.send(cmd, speeds)
        t0 = t_prev = self.now()
        wall_start = time.monotonic()
        while rclpy.ok():
            time.sleep(PERIOD)
            if self.cancelled():
                return False, 0.0, 'cancelled'
            if time.monotonic()-wall_start > max(30.0, max_time*10):
                return False, 0.0, 'clock_watchdog'
            q = dict(zip(names, self.joints(names)))
            err = {n: goal[n] - q[n] for n in names}
            max_err = max(abs(e) for e in err.values())
            if feedback:
                feedback(max_err)
            if all(abs(err[n]) < tols[n] for n in names):
                return True, max_err, "ok"
            t = self.now()
            if t - t0 > max_time:
                for n in names:  # blocked by something, maybe: the next move must not start by pushing harder
                    if abs(err[n]) > tols[n]:
                        self.compensation.pop(n, None)
                return False, max_err, "timeout"
            if t - t0 > duration:  # the profile has arrived: what is left is droop
                short = [n for n in names if abs(err[n]) > tols[n]]
                for n in short:
                    extra = cmd[n] - goal[n] + COMPENSATION_RATE * err[n] * (t - t_prev)
                    extra = max(-MAX_COMPENSATION, min(MAX_COMPENSATION, extra))
                    cmd[n] = clip(n, goal[n] + extra)
                    self.compensation[n] = extra
                if short:
                    self.send(cmd, {n: MAX_SPEED for n in names})
            t_prev = t
        return False, 0.0, "shutdown"

    def follow(self, names, waypoints, dt, tol, max_time, held=()):
        """Joint-space target, or closed-loop segments along a path.

        Streaming waypoints open-loop lets the arm cut the corner into the body. Each segment is short
        enough that the straight joint move stays on the collision-free path. max_time (sim s, 0 = no limit
        beyond each segment's own) bounds the whole path, not each segment.
        """
        deadline = self.now() + max_time if max_time else None
        if dt is None or len(waypoints) <= 1:
            return self.move_joints(names, waypoints[-1], tol=tol, max_time=max_time or 0.0, held=held)
        anchors = []
        prev = None
        for q in waypoints:
            if prev is None or max(abs(a - b) for a, b in zip(prev, q)) >= PATH_STEP:
                anchors.append(q)
                prev = q
        if anchors[-1] != waypoints[-1]:
            anchors.append(waypoints[-1])
        self.get_logger().info(f"follow {len(anchors)} segments, {len(names)} joints")
        for i, q in enumerate(anchors):
            if self.cancelled():
                return False, 0.0, 'cancelled'
            left = 0.0
            if deadline is not None:
                left = deadline - self.now()
                if left <= 0.0:
                    self.get_logger().info(f"segment {i + 1}/{len(anchors)}: out of max_time {max_time:.1f} s")
                    return False, 0.0, "timeout"
            ok, err, msg = self.move_joints(names, q, tol=tol if i == len(anchors) - 1 else 0.08, max_time=left, held=held)
            if not ok:
                here = self.joints(names)
                short = [f"{n} {t - c:+.3f}" for n, t, c in zip(names, q, here)
                         if c is not None and abs(t - c) > tol]
                self.get_logger().info(f"segment {i + 1}/{len(anchors)} {msg} err {err:.3f}: {', '.join(short)}")
                return False, err, msg
        return True, 0.0, "ok"

    def cancelled(self):
        handle = getattr(self._execution, 'handle', None)
        return handle is not None and handle.is_cancel_requested

    def hold_measured(self, handle):
        """Replace outstanding setpoints on abort; never open a possibly loaded hand."""
        if hasattr(handle.request, 'hand'):
            side = handle.request.hand
            with self.lock:
                state = self.hand_state.get(side)
            if state is not None and self.now() - (state.header.stamp.sec + state.header.stamp.nanosec*1e-9) <= .2:
                msg = JointState()
                msg.name, msg.position = list(state.name), list(state.position)
                self.hand_cmd[side].publish(msg)
        else:
            with self.lock:
                q = {n:v for n,v in self.q.items() if n in JOINTS and math.isfinite(v)}
            if q:
                self.send(q,{n:MAX_SPEED for n in q})
                self.compensation.clear()

    def execute(self, handle, action, run):
        # All three action servers share the robot, including auto-raise of the other arm.
        with self._execution_lock:
            self._execution.handle = handle
            try:
                result = action.Result(success=False) if self.cancelled() else run()
                if self.cancelled() or not result.success:
                    self.hold_measured(handle)
                if self.cancelled():
                    result.success = False
                    if hasattr(result, 'message'):
                        result.message = 'cancelled'
                    handle.canceled()
                else:
                    handle.succeed()
                return result
            except Exception as exc:
                self.hold_measured(handle)
                handle.abort()
                self.get_logger().error(str(exc))
                return action.Result(success=False)
            finally:
                self._execution.handle = None

    def exec_move_joints(self, handle):
        def run():
            g = handle.request
            fb = MoveJoints.Feedback()
            def feedback(e):
                fb.max_error = float(e)
                handle.publish_feedback(fb)
            ok, err, msg = self.move_joints(list(g.joints), list(g.positions), g.speed, g.tol, g.max_time, feedback)
            return MoveJoints.Result(success=ok, max_error=float(err), message=msg)
        return self.execute(handle,MoveJoints,run)

    def exec_hand(self, handle):
        return self.execute(handle,Hand,lambda: self.hand(handle))

    def hand(self, handle):
        g = handle.request
        res = Hand.Result()
        if g.hand not in self.hand_cmd or g.preset not in HAND_PRESETS:
            res.success = False
            return res
        target = [float(v) for v in HAND_PRESETS[g.preset]]
        # 0 keeps the limit already set (Hand.action). 1.0 pushes the object out; grasps pass 0.3-0.5.
        if g.force > 0 and not self._set_hand_limit(g.hand, g.force):
            self.get_logger().warn(f"set_force {g.hand} {g.force:.2f} was not applied")
        current = self._finger_pos(g.hand)
        speed = HAND_CLOSE_SPEED if _closing(current, target) else 1.0
        if not self._set_hand_speed(g.hand, speed):
            self.get_logger().warn(f"set_speed {g.hand} {speed:.2f} was not applied")
        # Turn the thumb across before the other fingers move. One command does both, and the
        # fingers arrive while the thumb is still swinging.
        if _thumb_first(current, target):
            staged = list(current)
            staged[THUMB] = target[THUMB]
            self._command_fingers(g.hand, staged, (6,))
            if not self._wait_fingers(handle, g.hand, staged, HAND_STROKE_S / speed + 0.6):
                return Hand.Result(success=False)
            self.get_logger().info(f"hand {g.hand} thumb across, closing the rest")
        self._command_fingers(g.hand, target)
        if not self._wait_fingers(handle, g.hand, target, HAND_STROKE_S / speed + 0.6):
            return Hand.Result(success=False)
        with self.lock:
            st = self.hand_state[g.hand]
        if st is not None and len(st.position) >= 6:
            res.position = [float(p) for p in st.position[:6]]
            res.effort = [float(e) for e in st.effort[:6]] if len(st.effort) >= 6 else []
            closing = [i for i in range(6) if target[i] < 0.9]
            # a closing finger stopped well short of its target with noticeable effort -> something is in the hand
            res.object_detected = any(
                res.position[i] > target[i] + 0.05 and (not res.effort or res.effort[i] > 0.1) for i in closing)
        res.success = (st is not None and len(st.position) >= 6 and
                       0 <= self.now()-(st.header.stamp.sec+st.header.stamp.nanosec*1e-9) <= .2)
        return res

    def _finger_pos(self, side):
        with self.lock:
            st = self.hand_state[side]
        if st is None or len(st.position) < 6:
            return None
        return [float(p) for p in st.position[:6]]

    def _command_fingers(self, side, positions, actuators=range(1, 7)):
        msg = JointState()
        msg.name = [str(i) for i in actuators]
        msg.position = [float(positions[i - 1]) for i in actuators]
        self.hand_cmd[side].publish(msg)

    def _set_hand_limit(self, side, force):
        req = SetForce.Request()
        for i in range(6):
            setattr(req, f"force{i}_ratio", float(force))
        res = self._call_hand(self.hand_force[side], req)
        return res is not None and res.force_accepted

    def _set_hand_speed(self, side, speed):
        req = SetSpeed.Request()
        for i in range(6):
            setattr(req, f"speed{i}_ratio", float(speed))
        res = self._call_hand(self.hand_speed[side], req)
        return res is not None and res.speed_accepted

    def _call_hand(self, client, request):
        """Service calls from the action callback. The executor is multi-threaded, so sleep until the reply."""
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=1.0):
            return None
        fut = client.call_async(request)
        deadline = time.perf_counter() + 1.0
        while rclpy.ok() and not fut.done() and time.perf_counter() < deadline:
            time.sleep(PERIOD)
        if not fut.done():
            return None
        try:
            return fut.result()
        except Exception as e:
            self.get_logger().warn(f"hand service failed: {e}")
            return None

    def _wait_fingers(self, handle, side, target, timeout):
        """Until the fingers sit near `target`, or stop short of it. Stillness before anything moves does not count.

        False if the goal was cancelled or the sim clock stalled (20 s wall), True otherwise."""
        t0, last, still_since, moved = self.now(), None, None, False
        wall_start = time.monotonic()
        fb = Hand.Feedback()
        while rclpy.ok() and self.now() - t0 < timeout:
            time.sleep(PERIOD)
            if self.cancelled() or time.monotonic() - wall_start > 20:
                return False
            pos = self._finger_pos(side)
            if pos is None:
                continue
            fb.position = pos
            handle.publish_feedback(fb)
            if last is not None and max(abs(a - b) for a, b in zip(pos, last)) >= 0.005:
                moved = True
                still_since = None
            elif last is not None:
                still_since = still_since or self.now()
            last = pos
            near = max(abs(pos[i] - target[i]) for i in range(6)) < FINGER_MOVE
            if still_since is None:
                continue
            held = self.now() - still_since
            if near and held > 0.15:
                return True
            if moved and held > 0.2:  # stopped on the way: the finger met the object
                return True
        return True

    # ---------------------------------------------------------------- MoveTcp
    def exec_move_tcp(self, handle):
        return self.execute(handle,MoveTcp,lambda: self.move_tcp(handle.request))

    def _planner_obj(self):
        if self._planner is None:
            from .curobo_planner import CuroboPlanner
            share = os.path.join(get_package_share_directory("hack_motion"), "robot")
            self._planner = CuroboPlanner(
                os.path.join(share, "tiangong2pro_with_hands.urdf"),
                os.path.join(share, "tienkung_curobo.yml"))
        self._planner.table, self._planner.trays = self._world()
        return self._planner

    def _wait_joints(self, timeout=5.0):
        """The action server is up before the first /joint_states. Wall clock: sim time may not be ticking yet."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if all(v is not None for v in self.joints(["shoulder_roll_l_joint", "shoulder_roll_r_joint"])):
                return True
            time.sleep(PERIOD)
        return False

    def _leave_colliding_home(self, planner):
        """Bring an arm that hangs at home up in front of the chest by joints (robot.RAISE_STEPS).

        At home the forearm spheres are inside the thigh, and cuRobo will not plan from a state in collision.
        Only when every joint of the arm is near home: from a working pose the same joint moves could sweep
        the arm through the table. Once the arm is out of the thigh, a step that would enter an object is
        skipped and the plan that follows goes around it.
        """
        for arm, names in ARM_JOINTS.items():
            q = self.joints(names)
            if any(v is None for v in q) or max(abs(v) for v in q) >= HOME_TOL:
                continue
            self.get_logger().info(f"{arm} arm at home, raising it in front of the chest")
            for i, step in enumerate(raise_steps(arm)):
                with self.lock:
                    snapshot = dict(self.q)
                try:
                    blocked = planner.segment_blocked(snapshot, names, step)
                except Exception:
                    self.get_logger().error(traceback.format_exc())
                    blocked = False
                if blocked:
                    self.get_logger().info(
                        f"{arm} raise stops before step {i + 1}: an object is on that joint path")
                    break
                ok, err, msg = self.move_joints(names, step, tol=0.1)
                if not ok:
                    raise RuntimeError(f"{arm} arm stopped on raise step {i + 1}: {msg} (err {err:.3f})")

    def tcp_error(self, arm, target):
        """Position (m) and rotation (rad) of the TCP tf against the goal, or (None, None)."""
        tf = None
        for _ in range(20):
            try:
                tf = self.tf_buffer.lookup_transform("pelvis", TCP_LINK[arm], rclpy.time.Time())
                break
            except Exception:
                time.sleep(0.05)
        if tf is None:
            return None, None
        t = tf.transform.translation
        r = tf.transform.rotation
        p = target.position
        o = target.orientation
        pos = math.sqrt((t.x - p.x) ** 2 + (t.y - p.y) ** 2 + (t.z - p.z) ** 2)
        dot = min(1.0, abs(r.x * o.x + r.y * o.y + r.z * o.z + r.w * o.w))
        return pos, 2.0 * math.acos(dot)

    def move_tcp(self, g):
        if g.arm not in ARM_JOINTS:
            return MoveTcp.Result(success=False, message=f"unknown arm {g.arm}")
        backend = self.get_parameter("planner").value
        if backend != "curobo":
            return MoveTcp.Result(success=False, message=f"not_implemented: planner {backend}")
        mode = g.mode or "direct"
        if mode not in ("direct", "linear", "ready"):
            return MoveTcp.Result(success=False, message=f"unknown mode {mode}")
        pos_tol = g.pos_tol or 0.01
        rot_tol = g.rot_tol or 0.1
        with self._tcp_lock:
            try:
                if not self._wait_joints():
                    return MoveTcp.Result(success=False, message="no /joint_states")
                planner = self._planner_obj()
                if planner.table is None:
                    # Without it the world has no table, and a plan goes through it. brain looks before it moves.
                    return MoveTcp.Result(success=False, message="no table plane from perception yet")
                with self.lock:
                    snapshot = dict(self.q)
                    planner.objects = self.scene_objects
                note = planner.load(snapshot)
                if note != "curobo already loaded":
                    self.get_logger().info(note)
                # Every object, including the one this goal will grasp. The raise is not aimed at it yet.
                planner.sync_world(None)
                self._leave_colliding_home(planner)
                planner.table, planner.trays = self._world()
                with self.lock:
                    snapshot = dict(self.q)
                    planner.objects = self.scene_objects
                names, waypoints, dt = planner.plan(
                    g.arm, g.target, snapshot, mode, pos_tol=pos_tol, rot_tol=rot_tol)
            except Unreachable as e:
                return MoveTcp.Result(success=False, message=f"unreachable: {e}")
            except Exception as e:
                self.get_logger().error(traceback.format_exc())
                return MoveTcp.Result(success=False, message=f"planner: {e}")
            if self.cancelled():
                return MoveTcp.Result(success=False, message='cancelled')
            known = [(n, i) for i, n in enumerate(names) if n in JOINTS]
            names = [n for n, _ in known]
            waypoints = [[q[i] for _, i in known] for q in waypoints]
            self.get_logger().info(f"move_tcp {g.arm} {mode}: {len(waypoints)} waypoints, {len(names)} joints, "
                                   f"start moved {planner.start_moved:.3f} rad, {planner.scene_note}")
            other = next(joints for arm, joints in ARM_JOINTS.items() if arm != g.arm)
            ok, err, msg = self.follow(names, waypoints, dt, tol=0.01, max_time=g.max_time, held=other)
        if mode == "ready":  # the goal is joints, the target pose is not used
            reached = ok or (msg == "timeout" and err <= READY_TOL)
            note = f" ({planner.ready_note})" if planner.ready_note else ""
            self.get_logger().info(f"move_tcp {g.arm} ready{note}: joint error {err:.3f} rad, {msg}")
            return MoveTcp.Result(success=reached, message="ok" if reached else msg)
        pos_err, rot_err = self.tcp_error(g.arm, g.target)
        if pos_err is None:
            return MoveTcp.Result(success=False, pos_error=0.0, rot_error=0.0, message=f"{msg}; no tf")
        # The goal is the TCP pose: a joint stalled a few mrad short must not fail a TCP that is in tolerance.
        reached = (ok or msg == "timeout") and pos_err <= pos_tol and rot_err <= rot_tol
        if reached:
            message = "ok"
        elif ok:
            message = "tolerance"
        else:
            message = msg
        self.get_logger().info(
            f"move_tcp {g.arm} {mode}: tcp {pos_err * 100:.2f} cm, {math.degrees(rot_err):.1f} deg, {message}")
        return MoveTcp.Result(success=reached, pos_error=float(pos_err), rot_error=float(rot_err), message=message)


def main():
    rclpy.init()
    node = MotionNode()
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
