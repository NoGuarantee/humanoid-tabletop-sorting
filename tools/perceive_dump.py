"""Why perception sees what it sees: one head frame through table_clusters, step by step.

Prints the table plane, the camera pose, the arm segments and every cluster with its rejection reason. Run beside
the sim with perception stopped (it starts its own PerceptionNode):

    python3 tools/perceive_dump.py [--camera head]
"""
import argparse
import threading
import time

import numpy as np
import rclpy
from hack_perception import table_clusters as tc
from hack_perception.perception_node import PerceptionNode
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="head")
    args = ap.parse_args()
    rclpy.init()
    node = PerceptionNode()
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    f = None
    for _ in range(100):
        f = node.snapshot(args.camera)
        if f is not None and node.get_clock().now().nanoseconds > 0:
            break
        time.sleep(0.1)
    time.sleep(1.0)
    f = node.snapshot(args.camera)
    cloud, m = node.cloud_in_pelvis(f)
    print("camera in pelvis:", np.round(m[:3, 3], 3), "view axis:", np.round(m[:3, 2], 3), flush=True)
    pts = cloud.reshape(-1, 3)
    ok = np.isfinite(pts[:, 2])
    print(f"points {ok.sum()} of {len(pts)}; x {np.nanpercentile(pts[:, 0], [5, 50, 95]).round(3)} "
          f"y {np.nanpercentile(pts[:, 1], [5, 50, 95]).round(3)} z {np.nanpercentile(pts[:, 2], [5, 50, 95]).round(3)}",
          flush=True)
    plane = tc.fit_table_plane(pts[ok])
    if plane is None:
        print("no table plane", flush=True)
    else:
        print(f"plane z at ref {plane.z_at(*tc.TABLE_REF_XY):.3f} normal {np.round(plane.normal, 3)}", flush=True)
        segs = node.arm_segments(f["depth"].header.stamp)
        print("arm segments:", [tuple(np.round(a, 2)) + tuple(np.round(b, 2)) for a, b in segs], flush=True)
        for label, robot in (("with arm mask", segs), ("without arm mask", ())):
            cs = tc.clusters_above(cloud, plane, robot)
            print(f"{label}: {len(cs)} clusters", flush=True)
            for c in cs:
                print(f"  at ({c.center[0]:.3f},{c.center[1]:.3f}) size {np.round(c.size, 3)} n {c.n_points} "
                      f"min_x {c.min_x:.3f} lowest {c.lowest:.3f} -> {tc.rejection(c) or 'object'}", flush=True)
    ex.shutdown()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
