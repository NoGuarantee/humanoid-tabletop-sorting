"""How far /tf trails the camera frames: latest stamps of /clock, /tf and head depth, once a second.

perception looks the camera up in /tf at the depth stamp, and twice on 2026-11-07 /tf was 0.2 s and 1.07 s
behind it. Run beside the sim:  python3 tools/stamp_probe.py [--seconds 10]
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage


def sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Probe(Node):
    def __init__(self):
        super().__init__("stamp_probe")
        self.clock = self.tf = self.depth = None
        self.tf_count = 0
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "clock", sec(m.clock)), 10)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 100)
        self.create_subscription(Image, "/cam_head/depth/image_raw",
                                 lambda m: setattr(self, "depth", sec(m.header.stamp)), 2)

    def on_tf(self, msg):
        self.tf_count += 1
        self.tf = max(sec(t.header.stamp) for t in msg.transforms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    rclpy.init()
    node = Probe()
    end = time.monotonic() + args.seconds
    next_print = time.monotonic() + 1.0
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.monotonic() >= next_print:
            next_print += 1.0
            c, t, d = node.clock, node.tf, node.depth
            if None in (c, t, d):
                print(f"clock {c} tf {t} depth {d}", flush=True)
                continue
            print(f"clock {c:.2f}  tf {t - c:+.3f}  depth {d - c:+.3f}  tf-depth {t - d:+.3f}  "
                  f"tf msgs/s {node.tf_count}", flush=True)
            node.tf_count = 0
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
