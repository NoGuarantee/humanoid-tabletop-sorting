"""Where the finger links really are in the TCP frame, from /tf, for each hand preset.

The grasp planner aims pads by FLAT_INDEX / FLAT_THUMB (grasp_planner.py), taken from the URDF. A leaned pinch that
closed beside the ladybug (2026-11-07) raised the question whether they match the simulated hand. Run with motion up
and the arm anywhere off the table:

    python3 tools/hand_tf_probe.py [--arm right] [--presets pinch_ready open pinch] [--raw 1,1,1,0.4,1,0.1]

Finding (2026-11-07): /tf does not show where the simulated fingers grip. It puts the right thumb tip of
pinch_ready at TCP (0.072, -0.042, -0.137) and curls the index toward -y; the planner has the thumb at
(0.037, +0.091, -0.140). The cartridge aimed between the /tf pads, TCP (0.045, -0.02), closed on air, while the
planner's midpoint (0.033, +0.057) lifted it three times out of three. Trust the pick runs, not this table.
"""
import argparse
import math
import threading
import time

import rclpy
from hack_brain.blocking import BlockingClients as Clients
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

LINKS = ("index_1", "index_2", "middle_2", "thumb_1", "thumb_2", "thumb_3", "thumb_4")
# Tips in the distal link frames, estimated from the URDF: the link's inertial origin lies about halfway out along
# the segment, which runs like the joint offset into it (index_1 -> index_2 along local +y, thumb_3 -> thumb_4
# along (0.66, 0.75, 0)). There are no fingertip meshes or collision spheres in this repo.
TIPS = {"index_2": (0.006, 0.040, -0.005), "thumb_4": (0.013, 0.019, -0.006)}


def rotate(q, v):
    """Rotate vector v by quaternion q (x, y, z, w)."""
    x, y, z, w = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)


class Probe(Node):
    def __init__(self):
        super().__init__("hand_tf_probe", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.api = Clients(self)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)

    def point(self, tcp, link, offset=(0.0, 0.0, 0.0)):
        t = self.tf.lookup_transform(tcp, link, Time(), timeout=Duration(seconds=2.0))
        q = t.transform.rotation
        p = t.transform.translation
        o = rotate((q.x, q.y, q.z, q.w), offset)
        return p.x + o[0], p.y + o[1], p.z + o[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="right")
    ap.add_argument("--presets", nargs="+", default=["pinch_ready", "open", "pinch"])
    ap.add_argument("--raw", nargs="*", default=[], metavar="A1,..,A6",
                    help="raw actuator targets (1 open .. 0 closed), published to /inspire_hand/ctrl")
    args = ap.parse_args()
    rclpy.init()
    node = Probe()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    while node.get_clock().now().nanoseconds == 0:
        time.sleep(0.1)
    node.api.wait_ready(timeout=120.0)
    side = args.arm
    tcp = f"{side}_tcp_link"
    ctrl = node.create_publisher(JointState, f"/inspire_hand/ctrl/{side}_hand", 10)
    for preset in args.presets + args.raw:
        if "," in preset:
            msg = JointState(name=[str(i) for i in range(1, 7)], position=[float(v) for v in preset.split(",")])
            ctrl.publish(msg)
            reading = "raw"
        else:
            hr = node.api.hand(side, preset)
            reading = " ".join(f"{v:.2f}" for v in hr.position) if hr is not None and hr.position else "-"
        t0 = node.get_clock().now()
        while (node.get_clock().now() - t0).nanoseconds < 2.0e9:
            time.sleep(0.05)
        print(f"== {preset}  reading {reading}", flush=True)
        for name in LINKS:
            try:
                x, y, z = node.point(tcp, f"{side}_{name}")
                print(f"  {name:9s} ({x:+.4f}, {y:+.4f}, {z:+.4f})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:9s} {e}", flush=True)
        for name, offset in TIPS.items():
            x, y, z = node.point(tcp, f"{side}_{name}", offset)
            print(f"  {name} tip ({x:+.4f}, {y:+.4f}, {z:+.4f})", flush=True)
    for x, y, z in ((0.0294, 0.0220, -0.2088), (0.0365, 0.0913, -0.1401)):
        print(f"planner pad ({x:+.4f}, {y:+.4f}, {z:+.4f}) |r| {math.sqrt(x*x+y*y+z*z):.4f}", flush=True)
    ex.shutdown()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
