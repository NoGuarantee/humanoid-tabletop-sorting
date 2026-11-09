#!/usr/bin/env python3
"""One move beside a real object, with that object in the cuRobo world. Needs the simulator.

The geometric probe (object_scene_probe.py --plan) never touches physics. This one does: perception
publishes the table, motion plans around those cuboids, and the wrist force says whether the arm
still pushed something. A clean move on this robot stays under about 10 N; the old low goals, aimed
through an object the planner did not know about, peaked at 36-101 N.

Head down and one scan while the arms still hang: hands over the table hide it, and a later scan would
replace the scene. Then the arms go out to the side by joints, and a top-down goal sits just outside
the nearest whole object, fingertips 3 cm above its top. The old contact point
(0.40, -0.35, 0.10) is reported too, and is not the pass condition: if it lands on an object the
planner leaves that object out on purpose.

  bash tools/remote.sh dev --timeout 1800 -- 'source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install && source install/setup.bash && \
    { pkill -f "[h]ack_motion/motion_node"; pkill -f "[h]ack_perception/perception_node"; sleep 2; \
      ros2 run hack_motion motion_node --ros-args -p use_sim_time:=true > /tmp/motion_avoid.log 2>&1 & \
      ros2 run hack_perception perception_node --ros-args -p use_sim_time:=true > /tmp/perception_avoid.log 2>&1 & \
      for i in $(seq 60); do ros2 action list | grep -q motion/move_tcp && \
        ros2 service list | grep -q perception/perceive && break; sleep 2; done; \
      python3 tools/object_avoid_check.py; status=$?; \
      echo --- motion ---; grep -E "perception scene|move_tcp|trays in the world" /tmp/motion_avoid.log | tail -30; \
      pkill -f "[h]ack_motion/motion_node"; pkill -f "[h]ack_perception/perception_node"; exit $status; }'
"""
import math
import threading
import time

import rclpy
from geometry_msgs.msg import WrenchStamped
from hack_brain.grasp_planner import pose, top_down
from hack_interfaces.action import MoveJoints, MoveTcp
from hack_interfaces.srv import Perceive
from hack_motion.robot import ARM_JOINTS, FINGERTIP, raise_steps
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

HEAD = 0.4
# The goal that previously drove through an object: 101 N with things on the table, 5 N with them removed.
OLD = ("right", 0.40, -0.35, 0.10)
CLEAR = 0.08  # m past the bounding radius, so the fingertips are not over the object
ABOVE = 0.03  # m, fingertips above the measured top
PUSH = 20.0  # N, above this the wrist hit something


def wait(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout)


class Check(Node):
    def __init__(self):
        super().__init__("object_avoid_check", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.tcp = ActionClient(self, MoveTcp, "motion/move_tcp")
        self.joints = ActionClient(self, MoveJoints, "motion/move_joints")
        self.perceive = self.create_client(Perceive, "perception/perceive")
        self.peak = {"left": 0.0, "right": 0.0}
        for arm in self.peak:
            self.create_subscription(WrenchStamped, f"/arm_6dof_{arm}",
                                     lambda msg, a=arm: self.on_wrench(a, msg), 10)

    def on_wrench(self, arm, msg):
        f = msg.wrench.force
        self.peak[arm] = max(self.peak[arm], math.sqrt(f.x ** 2 + f.y ** 2 + f.z ** 2))

    def take_peak(self):
        out = (self.peak["left"], self.peak["right"])
        self.peak = {"left": 0.0, "right": 0.0}
        return out

    def move(self, arm, target, mode, timeout):
        if not self.tcp.wait_for_server(timeout_sec=60.0):
            return None
        fut = self.tcp.send_goal_async(MoveTcp.Goal(arm=arm, target=target, mode=mode))
        if not wait(fut, 15.0) or not fut.result().accepted:
            return None
        res = fut.result().get_result_async()
        if not wait(res, timeout):
            return None
        return res.result().result

    def joints_to(self, names, positions):
        if not self.joints.wait_for_server(timeout_sec=30.0):
            return False
        fut = self.joints.send_goal_async(MoveJoints.Goal(
            joints=list(names), positions=[float(p) for p in positions]))
        if not wait(fut, 10.0) or not fut.result().accepted:
            return False
        res = fut.result().get_result_async()
        return wait(res, 90.0) and res.result().result.success

    def tilt(self):
        if not self.joints.wait_for_server(timeout_sec=30.0):
            return False
        fut = self.joints.send_goal_async(MoveJoints.Goal(joints=["head_pitch_joint"], positions=[HEAD]))
        if not wait(fut, 10.0) or not fut.result().accepted:
            return False
        res = fut.result().get_result_async()
        return wait(res, 60.0) and res.result().result.success

    def scene(self):
        """The table, retrying when the depth stamp is still ahead of /tf (the head just moved)."""
        if not self.perceive.wait_for_service(timeout_sec=30.0):
            return None
        last = "no answer"
        for _ in range(8):
            fut = self.perceive.call_async(Perceive.Request(camera="head", region="table"))
            if not wait(fut, 30.0):
                last = "timeout"
            else:
                res = fut.result()
                if res.success:
                    return res.scene
                last = res.message
            time.sleep(0.5)
        print(f"perceive: {last}", flush=True)
        return None


def beside(obj):
    """Top-down TCP just outside the object's bounding circle, fingertips ABOVE cm over the top."""
    radius = 0.5 * math.hypot(obj.size.x, obj.size.y)
    side = -1.0 if obj.center.y < 0 else 1.0
    x, y = obj.center.x, obj.center.y + side * (radius + CLEAR)
    arm = "left" if y > 0 else "right"
    return arm, pose(x, y, obj.top_z + ABOVE + FINGERTIP, top_down(0.0)), (x, y)


def report(label, result, peak, wall):
    left, right = peak
    if result is None:
        print(f"FAIL {label}  no result  L {left:.0f} R {right:.0f} N  wall {wall:.0f}s", flush=True)
        return False
    print(f"{'OK  ' if result.success else 'FAIL'} {label}  {result.message}  "
          f"tcp {result.pos_error * 100:.1f} cm  L {left:.0f} R {right:.0f} N  wall {wall:.0f}s", flush=True)
    return result.success


def main():
    rclpy.init()
    node = Check()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    code = 0
    try:
        if not node.tilt():
            print("FAIL head did not tilt", flush=True)
            return 1
        scene = node.scene()
        if scene is None or not scene.objects:
            print("FAIL no objects", flush=True)
            return 1
        whole = [o for o in scene.objects if not o.partially_visible] or list(scene.objects)
        for o in whole:
            print(f"object {o.track_id or '?'} ({o.center.x:.3f},{o.center.y:.3f}) "
                  f"size {o.size.x:.3f}x{o.size.y:.3f}x{o.size.z:.3f} top {o.top_z:.3f}", flush=True)
        obj = min(whole, key=lambda o: o.center.x ** 2 + o.center.y ** 2)
        arm, target, xy = beside(obj)
        print(f"beside {obj.track_id or '?'} -> {arm} ({xy[0]:.3f},{xy[1]:.3f},{target.position.z:.3f})", flush=True)
        # A cartesian plan from the hanging pose raises the arms by joints, and that path is not
        # collision-checked. Out to the side first. The scan already happened and must not be repeated:
        # an empty one would replace the latched scene.
        for side, names in ARM_JOINTS.items():
            for i, step in enumerate(raise_steps(side)[:2]):
                t0 = time.perf_counter()
                ok = node.joints_to(names, step)
                print(f"{'OK  ' if ok else 'FAIL'} aside {side} {i + 1}  "
                      f"wall {time.perf_counter() - t0:.0f}s", flush=True)
                if not ok:
                    return 1
        time.sleep(1.0)  # the latched scene has to reach motion before the plan
        node.take_peak()
        t0 = time.perf_counter()
        result = node.move(arm, target, "direct", timeout=240.0)
        peak = node.take_peak()
        wall = time.perf_counter() - t0
        ok = report("beside", result, peak, wall)
        pushed = max(peak) > PUSH
        if not ok or pushed:
            code = 1
        t0 = time.perf_counter()
        old = node.move(OLD[0], pose(OLD[1], OLD[2], OLD[3], top_down(0.0)), "direct", timeout=240.0)
        report(f"old ({OLD[1]:.2f},{OLD[2]:.2f},{OLD[3]:.2f})", old, node.take_peak(), time.perf_counter() - t0)
        if pushed:
            print(f"FAIL wrist force over {PUSH:.0f} N on the way past the object", flush=True)
        elif not ok:
            print("FAIL no plan beside the object", flush=True)
        else:
            print("passed", flush=True)
    finally:
        executor.shutdown()
        spin.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
