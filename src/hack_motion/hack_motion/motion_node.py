"""Motion server: actions MoveJoints, Hand, MoveTcp. Owner: motion.

MoveJoints: synchronised arrival of all joints, closed loop on /joint_states against the PD undershoot.
Hand:       named presets, reports whether the fingers stopped on an object.
MoveTcp:    cuRobo (parameter `planner`, default curobo). `direct` is one IK solution and a collision-free
            joint-space path to it, `linear` is a Cartesian trajectory. The reported error is the TF of
            the TCP, not the joint error.
"""
import math
import os
import threading
import time
import traceback

import rclpy
from ament_index_python.packages import get_package_share_directory
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from hack_interfaces.action import Hand, MoveJoints, MoveTcp
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from .curobo_planner import Unreachable
from .robot import ARM_JOINTS, HAND_PRESETS, JOINTS, MAX_SPEED, TCP_LINK, clip, raise_steps

PERIOD = 0.05  # s, real time between control checks
# 1/s, integrates the PD undershoot. The arm actuators settle with tau = kd/kp = 0.25 s, so this is damping 0.7.
COMPENSATION_RATE = 2.0
MAX_COMPENSATION = 0.3  # rad added on top of the target at most
HOME_TOL = 0.1  # rad, every arm joint within this of 0 -> the arm hangs at home
HELD_TOL = 0.05  # rad, the arm MoveTcp did not ask for: it only holds its pose


class MotionNode(Node):
    def __init__(self):
        super().__init__("hack_motion")
        self.lock = threading.Lock()
        self.q = {}
        self.compensation = {}  # joint -> rad on top of the target, kept between moves (see move_joints)
        self.hand_state = {"left": None, "right": None}
        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10, callback_group=cb)
        for side in ("left", "right"):
            self.create_subscription(JointState, f"/inspire_hand/state/{side}_hand",
                                     lambda m, s=side: self.on_hand(s, m), 10, callback_group=cb)
        self.cmd = {g: self.create_publisher(CmdSetMotorPosition, f"/{g}/cmd_pos", 10)
                    for g in ("arm", "head", "waist")}
        self.hand_cmd = {s: self.create_publisher(JointState, f"/inspire_hand/ctrl/{s}_hand", 10)
                         for s in ("left", "right")}
        ActionServer(self, MoveJoints, "motion/move_joints", self.exec_move_joints, callback_group=cb)
        ActionServer(self, Hand, "motion/hand", self.exec_hand, callback_group=cb)
        ActionServer(self, MoveTcp, "motion/move_tcp", self.exec_move_tcp, callback_group=cb)
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
        if not self._wait_joints(timeout=60.0):
            return
        with self._tcp_lock:
            try:
                with self.lock:
                    snapshot = dict(self.q)
                self.get_logger().info(self._planner_obj().load(snapshot))
            except Exception:
                self.get_logger().error(traceback.format_exc())

    # ---------------------------------------------------------------- state
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
        while rclpy.ok():
            time.sleep(PERIOD)
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
            if prev is None or max(abs(a - b) for a, b in zip(prev, q)) >= 0.3:
                anchors.append(q)
                prev = q
        if anchors[-1] != waypoints[-1]:
            anchors.append(waypoints[-1])
        self.get_logger().info(f"follow {len(anchors)} segments, {len(names)} joints")
        for i, q in enumerate(anchors):
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

    def exec_move_joints(self, handle):
        g = handle.request
        fb = MoveJoints.Feedback()

        def feedback(e):
            fb.max_error = float(e)
            handle.publish_feedback(fb)

        ok, err, msg = self.move_joints(list(g.joints), list(g.positions), g.speed, g.tol, g.max_time, feedback)
        handle.succeed()  # the action itself ran; whether the move reached the target is in the result
        return MoveJoints.Result(success=ok, max_error=float(err), message=msg)

    # ---------------------------------------------------------------- Hand
    def exec_hand(self, handle):
        res = self.hand(handle)
        handle.succeed()
        return res

    def hand(self, handle):
        g = handle.request
        res = Hand.Result()
        if g.hand not in self.hand_cmd or g.preset not in HAND_PRESETS:
            res.success = False
            return res
        if g.force > 0:
            # TODO(grasping): call /inspire_hand/set_force/{side}_hand (bodyctrl_msgs/srv/SetForce)
            self.get_logger().warn("Hand.force is not applied yet")
        target = HAND_PRESETS[g.preset]
        msg = JointState()
        msg.name = [str(i) for i in range(1, 7)]
        msg.position = [float(v) for v in target]
        self.hand_cmd[g.hand].publish(msg)

        t0, last, still_since = self.now(), None, None
        fb = Hand.Feedback()
        while rclpy.ok() and self.now() - t0 < 2.0:  # full stroke takes 0.8 s at default speed
            time.sleep(PERIOD)
            with self.lock:
                st = self.hand_state[g.hand]
            if st is None or len(st.position) < 6:
                continue
            pos = list(st.position[:6])
            fb.position = [float(p) for p in pos]
            handle.publish_feedback(fb)
            if last is not None and max(abs(a - b) for a, b in zip(pos, last)) < 0.005:
                still_since = still_since or self.now()
                if self.now() - still_since > 0.2:
                    break
            else:
                still_since = None
            last = pos
        with self.lock:
            st = self.hand_state[g.hand]
        if st is not None and len(st.position) >= 6:
            res.position = [float(p) for p in st.position[:6]]
            res.effort = [float(e) for e in st.effort[:6]] if len(st.effort) >= 6 else []
            closing = [i for i in range(6) if target[i] < 0.9]
            # a closing finger stopped well short of its target with noticeable effort -> something is in the hand
            res.object_detected = any(
                res.position[i] > target[i] + 0.05 and (not res.effort or res.effort[i] > 0.1) for i in closing)
        res.success = True
        return res

    # ---------------------------------------------------------------- MoveTcp
    def exec_move_tcp(self, handle):
        res = self.move_tcp(handle.request)
        handle.succeed()
        return res

    def _planner_obj(self):
        if self._planner is None:
            from .curobo_planner import CuroboPlanner
            share = os.path.join(get_package_share_directory("hack_motion"), "robot")
            self._planner = CuroboPlanner(
                os.path.join(share, "tiangong2pro_with_hands.urdf"),
                os.path.join(share, "tienkung_curobo.yml"))
        return self._planner

    def _wait_joints(self, timeout=5.0):
        """The action server is up before the first /joint_states. Wall clock: sim time may not be ticking yet."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if all(v is not None for v in self.joints(["shoulder_roll_l_joint", "shoulder_roll_r_joint"])):
                return True
            time.sleep(PERIOD)
        return False

    def _leave_colliding_home(self):
        """Bring an arm that hangs at home up in front of the chest by joints (robot.RAISE_STEPS).

        At home the forearm spheres are inside the thigh, and cuRobo will not plan from a state in collision.
        A cuRobo path from the hang would also swing the hand under the table edge: the table is not in the
        cuRobo world yet. Only when every joint of the arm is near home: these joint moves are not
        collision-checked, and from a working pose they could sweep the arm through the table.
        """
        for arm, names in ARM_JOINTS.items():
            q = self.joints(names)
            if any(v is None for v in q) or max(abs(v) for v in q) >= HOME_TOL:
                continue
            self.get_logger().info(f"{arm} arm at home, raising it in front of the chest")
            for i, step in enumerate(raise_steps(arm)):
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
        if mode not in ("direct", "linear"):
            return MoveTcp.Result(success=False, message=f"unknown mode {mode}")
        pos_tol = g.pos_tol or 0.01
        rot_tol = g.rot_tol or 0.1
        with self._tcp_lock:
            try:
                if not self._wait_joints():
                    return MoveTcp.Result(success=False, message="no /joint_states")
                self._leave_colliding_home()
                planner = self._planner_obj()
                with self.lock:
                    snapshot = dict(self.q)
                note = planner.load(snapshot)
                if note != "curobo already loaded":
                    self.get_logger().info(note)
                names, waypoints, dt = planner.plan(
                    g.arm, g.target, snapshot, mode, pos_tol=pos_tol, rot_tol=rot_tol)
            except Unreachable as e:
                return MoveTcp.Result(success=False, message=f"unreachable: {e}")
            except Exception as e:
                self.get_logger().error(traceback.format_exc())
                return MoveTcp.Result(success=False, message=f"planner: {e}")
            known = [(n, i) for i, n in enumerate(names) if n in JOINTS]
            names = [n for n, _ in known]
            waypoints = [[q[i] for _, i in known] for q in waypoints]
            self.get_logger().info(f"move_tcp {g.arm} {mode}: {len(waypoints)} waypoints, {len(names)} joints")
            other = next(joints for arm, joints in ARM_JOINTS.items() if arm != g.arm)
            ok, _, msg = self.follow(names, waypoints, dt, tol=0.01, max_time=g.max_time, held=other)
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
