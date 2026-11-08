#!/usr/bin/env python3
"""Pick one object from a known point and drop it in the tray on the same side. Milestone M1.

The other objects are parked through `/sim/objects/set` (a debug topic: this script lives in tools/).
`--object` is any substring of a name in classify.CATALOG, `--at` is along the table, y, yaw.
The default dish uses its placed pose: the head's detector drops it, its height sits in the tray band.
Anything else uses the head camera unless `--known` is given, and the difference from the placed pose
is printed. The placed pose in the pelvis frame comes from `on_table`, the same 4° pitch the trays use.
motion builds its world from perception/scene, so the script publishes the measured table and trays there
(tools/scene_measured.py) before the first move; a perceive replaces them with what the camera saw.
The tray is chosen by the arm (`y > 0` -> left), not by the task rule.
Open fingers meet the table at TCP z = 0.039 + 0.070 x (tools/table_touch.py); the script refuses a
vertical hand whose TCP is within 5 mm of that. A leaned pinch is checked in grasp_planner.
`--set key=value` overrides this object's grasp_db entry; `--sweep key=v1,v2,...` picks once per value
(placing the object again each time) and prints a table, e.g. `--set preset=pinch --sweep across=0.05,0.06`.
A value that holds belongs in grasp_db.yaml by_name.

Needs hack_motion and hack_perception, not hack_brain: brain would publish /sim/task/done and exit.
The dev container stays up and `kill` of `ros2 run` leaves its child alive, so stop the nodes by name.

  bash tools/remote.sh dev --timeout 2400 -- 'source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install && source install/setup.bash && \
    { pkill -f "[h]ack_motion/motion_node"; pkill -f "[h]ack_perception/perception_node"; sleep 2; \
      ros2 run hack_motion motion_node --ros-args -p use_sim_time:=true & \
      ros2 run hack_perception perception_node --ros-args -p use_sim_time:=true & \
      for i in $(seq 90); do ros2 action list | grep -q motion/move_tcp && \
        ros2 service list | grep -q perception/perceive && break; sleep 2; done; \
      python3 tools/pick_one.py --object Brother; status=$?; \
      pkill -f "[h]ack_motion/motion_node"; pkill -f "[h]ack_perception/perception_node"; exit $status; }'
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from hack_brain.blocking import BlockingClients as Clients
from hack_brain.grasp_planner import FINGERTIP, load_db, plan_grasps, plan_place
from hack_brain.skills import Skills
from hack_interfaces.msg import Container, DetectedObject, SceneObjects
from hack_motion.robot import TRAY_SIZE
from hack_perception.classify import CATALOG
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from scene_measured import TABLE, TRAYS, on_table, table_z, tray_rim, trays
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# Same look-at-table pose as brain_node.LOOK_TABLE. Arms then go to the ready pose, out of the picture.
LOOK = {"head_pitch_joint": 0.4, "head_yaw_joint": 0.0, "body_yaw_joint": 0.0}
# TCP height where open fingers start to meet the table, from tools/table_touch.py. Stay above it.
TOUCH_Z = (0.039, 0.070)
TOUCH_MARGIN = 0.005
# The flat dish of pool 4. (0.38, -0.18) put a corner inside tray_center (rim x 0.43..0.65,
# y -0.16..0.16) and the forearm lay across that tray. This spot is in front of the right arm:
# 8 cm clear of tray_center and 13 cm clear of tray_right.
DISH = "Threshold_Bamboo_Ceramic_Soap_Dish"
PLACE_AT = (0.34, -0.32, 0.0)  # along the table, y, yaw
# Metres of air between the object box and a tray. Less than this and the wrist sits on the rim.
TRAY_CLEAR = 0.04

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "hack_brain", "config", "grasp_db.yaml")


def wait_sim(node, seconds):
    t0 = node.get_clock().now().nanoseconds * 1e-9
    while rclpy.ok() and node.get_clock().now().nanoseconds * 1e-9 - t0 < seconds:
        time.sleep(0.05)


def touch_z(x):
    return TOUCH_Z[0] + TOUCH_Z[1] * x


def measured_scene():
    """SceneObjects with the measured table plane and the three trays, the way perception reports them."""
    scene = SceneObjects(table_z=float(table_z(0.0)), table_plane=[float(v) for v in TABLE])
    scene.header.frame_id = "pelvis"
    for ident, x, y, z, yaw, sx, sy, sz in trays():
        scene.containers.append(Container(
            id=ident, kind="tray", size=Vector3(x=sx, y=sy, z=sz),
            pose=Pose(position=Point(x=x, y=y, z=z),
                      orientation=Quaternion(z=math.sin(yaw / 2), w=math.cos(yaw / 2)))))
    return scene


def publish_measured_scene(node):
    """Give motion a table before the first move. Latched, as perception publishes it."""
    pub = node.create_publisher(
        SceneObjects, "perception/scene", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    scene = measured_scene()
    scene.header.stamp = node.get_clock().now().to_msg()
    pub.publish(scene)
    node.scene_pub = pub  # keep it alive: a latched message lives as long as its publisher


def place_object(node, name, at):
    """Park every object and put this one at `at`. Returns False if the sim never answers."""
    got = []
    node.create_subscription(
        String, "/sim/objects/names", lambda m: got.append(json.loads(m.data)),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    pub = node.create_publisher(String, "/sim/objects/set", 10)
    t0 = time.time()
    while not got and time.time() - t0 < 30.0:
        time.sleep(0.05)
    if not got or name not in got[0]:
        print(f"FAIL {name} is not in /sim/objects/names", flush=True)
        return False
    while pub.get_subscription_count() == 0 and time.time() - t0 < 30.0:
        time.sleep(0.05)
    if pub.get_subscription_count() == 0:
        print("FAIL nothing subscribed to /sim/objects/set", flush=True)
        return False
    layout = {n: "park" for n in got[0]}
    layout[name] = {"x": at[0], "y": at[1], "yaw": at[2], "center": True}
    pub.publish(String(data=json.dumps(layout)))
    print(f"placed {name} at ({at[0]:.2f},{at[1]:.2f}) yaw {at[2]:.2f}, parked {len(layout) - 1}", flush=True)
    return True


def tray_boxes():
    """Pelvis AABB of each tray, from the temporary collision sizes in robot.py."""
    hx, hy = TRAY_SIZE[0] / 2.0, TRAY_SIZE[1] / 2.0
    boxes = []
    for name, (along, y) in TRAYS.items():
        x0 = on_table(along - hx, y, 0.0)[0]
        x1 = on_table(along + hx, y, 0.0)[0]
        boxes.append((name, min(x0, x1), max(x0, x1), y - hy, y + hy))
    return boxes


def hits_tray(obj):
    """Tray name when the object box comes within TRAY_CLEAR of it."""
    hx, hy = obj.size.x / 2.0, obj.size.y / 2.0
    for name, x0, x1, y0, y1 in tray_boxes():
        if (obj.center.x - hx < x1 + TRAY_CLEAR and obj.center.x + hx > x0 - TRAY_CLEAR
                and obj.center.y - hy < y1 + TRAY_CLEAR and obj.center.y + hy > y0 - TRAY_CLEAR):
            return name
    return None


def known_object(name, at):
    """The placed object in the pelvis frame. Layout metres are along the table; the pelvis is pitched 4°."""
    sx, sy, sz, category, _ = CATALOG[name]
    cx, cy, cz = on_table(at[0], at[1], sz / 2.0)
    _, _, top = on_table(at[0], at[1], sz)
    obj = DetectedObject(
        track_id="known", name=name, category=category, confidence=1.0,
        center=Point(x=cx, y=cy, z=cz), yaw=float(at[2]),
        size=Vector3(x=sx, y=sy, z=sz), top_z=float(top))
    scene = SceneObjects(table_z=float(table_z(cx)))
    scene.objects.append(obj)
    return obj, scene


def seen_object(node, known):
    """The same object from the head camera, or None. Prints how far it is from the placed pose."""
    scene = node.api.perceive("head", "table")
    if scene is None:
        print("perception gave no scene", flush=True)
        return None, None
    for o in scene.objects:
        print(f"  seen {o.name or '?'} ({o.category} {o.confidence:.2f}) at ({o.center.x:.3f},{o.center.y:.3f}) "
              f"top {o.top_z:+.3f} size {o.size.x:.3f} {o.size.y:.3f} {o.size.z:.3f} yaw {o.yaw:.2f} "
              f"partial {o.partially_visible}", flush=True)
    near = [o for o in scene.objects
            if math.hypot(o.center.x - known.center.x, o.center.y - known.center.y) < 0.10]
    if not near:
        print("no cluster within 10 cm of the placed pose", flush=True)
        return None, None
    obj = max(near, key=lambda o: o.confidence)
    print(f"seen minus placed: x {obj.center.x - known.center.x:+.3f} y {obj.center.y - known.center.y:+.3f} "
          f"top {obj.top_z - known.top_z:+.3f}", flush=True)
    if obj.name != known.name:
        print(f"camera says {obj.name or '?'}, using the placed name {known.name}", flush=True)
        obj.name, obj.category = known.name, known.category
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
    if cand.leaned:
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
        self.skills = Skills(load_db(DB), self.api, self.get_logger())
        self.waist = None
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)

    def on_joints(self, msg):
        if "body_yaw_joint" in msg.name:
            self.waist = msg.position[msg.name.index("body_yaw_joint")]


def watch(node, stop, aim=0.0):
    """--watch: the head turns against the waist, so its camera stays on the object (bearing aim, rad from pelvis x)
    while a grasp swings the waist.

    The plans turn the waist up to 0.8 rad, and the video of a slip then shows the far end of the table. Motion
    serializes only TCP moves, so this runs beside the pick. Debugging only: the brain keeps the head still.
    """
    target = 0.0
    while not stop.is_set():
        w = node.waist
        if w is not None and abs(aim - w - target) > 0.05:
            target = max(-1.5, min(1.5, aim - w))
            node.api.move_joints(["head_yaw_joint", "head_pitch_joint"], [target, 0.6])
        time.sleep(0.2)


def run(node, name, at, use_camera, watching=False):
    while rclpy.ok() and node.get_clock().now().nanoseconds == 0:
        time.sleep(0.1)
    if not node.api.wait_ready(timeout=120.0):
        print("FAIL perception or motion is not up", flush=True)
        return 1
    publish_measured_scene(node)
    if not node.api.move_joints(list(LOOK), list(LOOK.values())):
        print("FAIL head did not look at the table", flush=True)
        return 1
    wait_sim(node, 0.5)
    # Both arms to the ready pose, out of the head camera's view. The first plan raises both hanging arms in front
    # of the chest, across the picture. A park by TCP pose ended by joint stops twice (shoulder -2.3 rad on the
    # left, elbow_yaw -2.95 on the right on 2026-11-07), and no pre-grasp planned from there.
    for arm in ("right", "left"):
        if not node.skills.ready(arm):
            print(f"FAIL could not bring {arm} to the ready pose", flush=True)
            return 1
    if not place_object(node, name, at):
        return 1
    wait_sim(node, 1.0)
    obj, scene = known_object(name, at)
    print(f"placed pose ({obj.center.x:.3f},{obj.center.y:.3f}) top {obj.top_z:+.3f} "
          f"table_z {scene.table_z:+.3f}", flush=True)
    if use_camera:
        obj, scene = seen_object(node, obj)
        if obj is None:
            print("FAIL the camera did not give the object", flush=True)
            return 1
    hit = hits_tray(obj)
    if hit is not None:
        print(f"FAIL the object is within {TRAY_CLEAR * 100:.0f} cm of {hit}", flush=True)
        return 1
    cand = plan_grasps(obj, scene, node.skills.db)[0]
    tray = tray_for(cand.arm)
    place = plan_place(cand.arm, tray, node.skills.db)
    if show_plan(obj, cand, place) < TOUCH_MARGIN:
        print("FAIL refusing: the fingertips would meet the table", flush=True)
        return 1
    print(f"grasp -> {tray.id}", flush=True)
    stop = threading.Event()
    if watching:
        aim = math.atan2(obj.center.y, obj.center.x)
        threading.Thread(target=watch, args=(node, stop, aim), daemon=True).start()
    try:
        held = node.skills.pick(obj, scene)
    finally:
        stop.set()
    if held is None:
        print("FAIL nothing held; hand should be open", flush=True)
        return 1
    print(f"holding with {held.arm}, placing into {tray.id}", flush=True)
    if not node.skills.place(held, tray):
        print("FAIL place did not finish; grip kept if the release was not reached", flush=True)
        return 1
    print(f"OK placed into {tray.id}", flush=True)
    return 0


def parse_kv(kv):
    """'across=0.05' -> ('across', 0.05); values that are not numbers stay strings."""
    key, value = kv.split("=", 1)
    try:
        return key, float(value)
    except ValueError:
        return key, value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default=DISH, help="substring of a CATALOG name")
    ap.add_argument("--at", type=float, nargs=3, default=PLACE_AT, metavar=("ALONG", "Y", "YAW"))
    ap.add_argument("--known", action="store_true", help="grasp from the placed pose, not the camera")
    ap.add_argument("--watch", action="store_true", help="turn the head against the waist during the pick (video)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="grasp_db parameter for this object, e.g. preset=pinch or across=0.05")
    ap.add_argument("--sweep", metavar="KEY=V1,V2,...",
                    help="pick once per value (object placed again each time), then print a table")
    args = ap.parse_args()
    names = [n for n in CATALOG if args.object in n]
    if len(names) != 1:
        sys.exit(f"--object {args.object} matches {names}")
    use_camera = names[0] != DISH and not args.known
    rclpy.init()
    node = PickOne()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    status = 1
    entry = node.skills.db.setdefault("by_name", {}).setdefault(names[0], {})
    entry.update(dict(parse_kv(kv) for kv in args.set))
    try:
        if args.sweep:
            key, values = args.sweep.split("=", 1)
            results = []
            for v in values.split(","):
                entry[key] = parse_kv(f"{key}={v}")[1]
                print(f"==== {key}={entry[key]}", flush=True)
                results.append((entry[key], run(node, names[0], tuple(args.at), use_camera, args.watch)))
            print(f"==== sweep {key} on {names[0]} with {args.set}", flush=True)
            for v, st in results:
                print(f"  {key}={v}: {'OK' if st == 0 else 'fail'}", flush=True)
            status = 0 if any(st == 0 for _, st in results) else 1
        else:
            status = run(node, names[0], tuple(args.at), use_camera, args.watch)
    finally:
        executor.shutdown()
        spin.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(status)


if __name__ == "__main__":
    main()
