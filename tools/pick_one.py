#!/usr/bin/env python3
"""Pick one soap dish from a known point and drop it in the tray on the same side. Milestone M1.

The other objects are parked through `/sim/objects/set` (a debug topic: this script lives in tools/).
The dish is placed at DISH_AT and the grasp uses that pose, not a camera cluster: the head's detector
drops this dish, its height sits in the tray band. The pose in the pelvis frame comes from `on_table`,
the same 4° pitch the trays use. The tray is chosen by the arm (`y > 0` -> left), not by the task rule.
Open fingers meet the table at TCP z = 0.039 + 0.070 x (tools/table_touch.py); the script refuses a
vertical hand whose TCP is within 5 mm of that. A leaned pinch is checked in grasp_planner.

Needs hack_motion and hack_perception, not hack_brain: brain would publish /sim/task/done and exit.
The dev container stays up and `kill` of `ros2 run` leaves its child alive, so stop the nodes by name.

  bash tools/remote.sh dev --timeout 2400 -- 'source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install && source install/setup.bash && \
    { pkill -f "[h]ack_motion/motion_node"; pkill -f "[h]ack_perception/perception_node"; sleep 2; \
      ros2 run hack_motion motion_node --ros-args -p use_sim_time:=true & \
      ros2 run hack_perception perception_node --ros-args -p use_sim_time:=true & \
      for i in $(seq 90); do ros2 action list | grep -q motion/move_tcp && \
        ros2 service list | grep -q perception/perceive && break; sleep 2; done; \
      python3 tools/pick_one.py; status=$?; \
      pkill -f "[h]ack_motion/motion_node"; pkill -f "[h]ack_perception/perception_node"; exit $status; }'
"""
import json
import os
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from hack_brain.clients import Clients
from hack_brain.grasp_planner import FINGERTIP, axis_z, load_db, plan_grasps, plan_place, pose, top_down
from hack_brain.skills import Skills
from hack_interfaces.msg import Container, DetectedObject, SceneObjects
from hack_motion.robot import on_table, table_z, tray_rim
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

# Same look-at-table pose as brain_node.LOOK_TABLE. Arms then go to goals move_tcp_check already reached,
# so the hanging hands are out of the cloud before the picture.
LOOK = {"head_pitch_joint": 0.4, "head_yaw_joint": 0.0, "body_yaw_joint": 0.0}
PARK = (("right", 0.40, -0.35, 0.30), ("left", 0.40, 0.35, 0.30))
# TCP height where open fingers start to meet the table, from tools/table_touch.py. Stay above it.
TOUCH_Z = (0.039, 0.070)
TOUCH_MARGIN = 0.005
# The flat dish of pool 4, on the right, clear of the parked hand at y -0.35 and of tray_right at y -0.69.
DISH = "Threshold_Bamboo_Ceramic_Soap_Dish"
DISH_AT = (0.38, -0.18, 0.0)  # along the table, y, yaw. Right of centre, clear of the parked hand and the tray.
DISH_SIZE = (0.125, 0.098, 0.051)  # mesh box, long / short / height, same as classify.CATALOG

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "hack_brain", "config", "grasp_db.yaml")


def wait_sim(node, seconds):
    t0 = node.get_clock().now().nanoseconds * 1e-9
    while rclpy.ok() and node.get_clock().now().nanoseconds * 1e-9 - t0 < seconds:
        time.sleep(0.05)


def touch_z(x):
    return TOUCH_Z[0] + TOUCH_Z[1] * x


def place_dish(node):
    """Park every object and put the soap dish at DISH_AT. Returns False if the sim never answers."""
    got = []
    node.create_subscription(
        String, "/sim/objects/names", lambda m: got.append(json.loads(m.data)),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    pub = node.create_publisher(String, "/sim/objects/set", 10)
    t0 = time.time()
    while not got and time.time() - t0 < 30.0:
        time.sleep(0.05)
    if not got or DISH not in got[0]:
        print(f"FAIL {DISH} is not in /sim/objects/names", flush=True)
        return False
    while pub.get_subscription_count() == 0 and time.time() - t0 < 30.0:
        time.sleep(0.05)
    if pub.get_subscription_count() == 0:
        print("FAIL nothing subscribed to /sim/objects/set", flush=True)
        return False
    layout = {name: "park" for name in got[0]}
    layout[DISH] = {"x": DISH_AT[0], "y": DISH_AT[1], "yaw": DISH_AT[2], "center": True}
    pub.publish(String(data=json.dumps(layout)))
    print(f"placed {DISH} at ({DISH_AT[0]:.2f},{DISH_AT[1]:.2f}) yaw {DISH_AT[2]:.2f}, "
          f"parked {len(layout) - 1}", flush=True)
    return True


def known_dish():
    """The placed dish in the pelvis frame. Layout metres are along the table; the pelvis is pitched 4°."""
    sx, sy, sz = DISH_SIZE
    cx, cy, cz = on_table(DISH_AT[0], DISH_AT[1], sz / 2.0)
    _, _, top = on_table(DISH_AT[0], DISH_AT[1], sz)
    obj = DetectedObject(
        track_id="dish", name=DISH, category="dish", confidence=1.0,
        center=Point(x=cx, y=cy, z=cz), yaw=float(DISH_AT[2]),
        size=Vector3(x=sx, y=sy, z=sz), top_z=float(top))
    scene = SceneObjects(table_z=float(table_z(cx)))
    scene.objects.append(obj)
    return obj, scene


def tray_for(arm):
    """Opening of the tray on this arm's side. sort2 and sort3 both have the side trays."""
    name = "tray_left" if arm == "left" else "tray_right"
    x, y, z = tray_rim(name)
    return Container(id=name, kind="tray",
                     pose=Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(w=1.0)))


def show_plan(obj, cand, place):
    g, tip = cand.grasp.position, cand.grasp.position.z - FINGERTIP
    margin = g.z - touch_z(obj.center.x)
    print(f"pick {obj.track_id or '?'} {cand.arm} xy ({obj.center.x:.3f},{obj.center.y:.3f}) "
          f"grasp ({g.x:.3f},{g.y:.3f}) "
          f"top {obj.top_z:+.3f} size {obj.size.x:.3f} {obj.size.y:.3f} {obj.size.z:.3f} "
          f"yaw {obj.yaw:.2f} preset {cand.hand_preset}", flush=True)
    print(f"  fingertips {tip:+.3f}  tcp grasp {g.z:+.3f}  pre {cand.pre_grasp.position.z:+.3f}  "
          f"lift {cand.lift.position.z:+.3f}", flush=True)
    print(f"  table contact at x {obj.center.x:.2f} is tcp {touch_z(obj.center.x):+.3f}, "
          f"margin {margin * 100:.1f} cm", flush=True)
    if axis_z(cand.grasp.orientation)[2] < 0.85:
        # The touch line is for open fingers straight down. A leaned pinch clears the table beside the object.
        print("  hand is leaned; the vertical touch line does not apply", flush=True)
        margin = TOUCH_MARGIN
    r = place.release.position
    print(f"place {place.arm} tcp release ({r.x:.3f},{r.y:.3f},{r.z:+.3f})  "
          f"approach {place.approach.position.z:+.3f}", flush=True)
    return margin


class PickOne(Node):
    def __init__(self):
        super().__init__("pick_one", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.api = Clients(self)
        self.skills = Skills(self.api, load_db(DB), self.get_logger())


def run(node):
    while rclpy.ok() and node.get_clock().now().nanoseconds == 0:
        time.sleep(0.1)
    if not node.api.wait_ready(timeout=120.0):
        print("FAIL perception or motion is not up", flush=True)
        return 1
    if not node.api.move_joints(list(LOOK), list(LOOK.values())):
        print("FAIL head did not look at the table", flush=True)
        return 1
    wait_sim(node, 0.5)
    q = top_down(0.0)
    for arm, x, y, z in PARK:
        if not node.api.move_tcp(arm, pose(x, y, z, q), "direct"):
            print(f"FAIL could not park {arm} above the table", flush=True)
            return 1
    if not place_dish(node):
        return 1
    wait_sim(node, 1.0)
    obj, scene = known_dish()
    print(f"dish at ({obj.center.x:.3f},{obj.center.y:.3f}) top {obj.top_z:+.3f} "
          f"table_z {scene.table_z:+.3f}", flush=True)
    cand = plan_grasps(obj, scene, node.skills.db)[0]
    tray = tray_for(cand.arm)
    place = plan_place(cand.arm, tray, node.skills.db)
    if show_plan(obj, cand, place) < TOUCH_MARGIN:
        print("FAIL refusing: the fingertips would meet the table", flush=True)
        return 1
    print(f"grasp -> {tray.id}", flush=True)
    held = node.skills.pick(obj, scene)
    if held is None:
        print("FAIL nothing held; hand should be open", flush=True)
        return 1
    print(f"holding with {held.arm}, placing into {tray.id}", flush=True)
    if not node.skills.place(held, tray):
        print("FAIL place did not finish; grip kept if the release was not reached", flush=True)
        return 1
    print(f"OK placed into {tray.id}", flush=True)
    return 0


def main():
    rclpy.init()
    node = PickOne()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    status = 1
    try:
        status = run(node)
    finally:
        executor.shutdown()
        spin.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(status)


if __name__ == "__main__":
    main()
