#!/usr/bin/env python3
"""Check a recorded robot motion for self-intersections, joint limits and speed, on the manufacturer's meshes.

The sim has self-collisions off and they stay off in grading, but the organisers analyse the trajectories separately:
an arm through the robot means the task does not count. Their method is secret, so this checks the real meshes
instead of the cuRobo spheres. Debug tool, nothing in src/ or run.sh uses it.

Record /joint_states during a run (needs ROS, stops on /sim/task/done or Ctrl-C):

  python3 tools/trajectory_self_check.py record motion.jsonl

Check it anywhere, no ROS or cuRobo. Needs `pip install numpy trimesh python-fcl` and the meshes of the pinned commit
(`python3 tools/fit_tienkung_spheres.py fetch` puts them in data/, the default for --meshes):

  python3 tools/trajectory_self_check.py check motion.jsonl [--margin 0.02]

A line of the file is {"t": sim seconds, "q": {joint: rad}}. Exit code 1 if a new intersection, a limit or a speed
problem was found. Limits of the method: meshes are surfaces, so a link entirely inside another is seen only as it
enters, and an object held in the hand is not modelled.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "src/hack_motion/robot/tiangong2pro_with_hands.urdf"
MESHES = ROOT / "data/tienkung_urdf/4c2c061d72dc"  # CACHE in fit_tienkung_spheres.py
MIN_STEP = 0.01  # rad, samples closer than this in every joint are not checked again
SPEED_WINDOW = 0.1  # sim s, joint state noise would dominate a shorter finite difference
SPEED_TOL = 1.1  # the limit in the URDF times this
LIMIT_TOL = 0.01  # rad


def record(path):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    rclpy.init()
    node = rclpy.create_node("trajectory_recorder")
    stop = []
    with open(path, "w") as out:
        def on_joints(msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            out.write(json.dumps({"t": t, "q": dict(zip(msg.name, msg.position))}) + "\n")

        node.create_subscription(JointState, "/joint_states", on_joints, qos_profile_sensor_data)
        node.create_subscription(Empty, "/sim/task/done", lambda _: stop.append(1), 10)
        try:
            while rclpy.ok() and not stop:
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            pass
    node.destroy_node()
    rclpy.try_shutdown()


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    import numpy as np

    cr, sr, cp, sp, cy, sy = np.cos(rpy[0]), np.sin(rpy[0]), np.cos(rpy[1]), np.sin(rpy[1]), np.cos(rpy[2]), np.sin(rpy[2])
    out = np.eye(4)
    out[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                   [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                   [-sp, cp * sr, cp * cr]]
    out[:3, 3] = xyz
    return out


def rotation(axis, angle):
    import numpy as np

    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    return out


def floats(text):
    return tuple(float(v) for v in text.split())


class Robot:
    """URDF kinematics and collision meshes: only what the check needs."""

    def __init__(self, urdf, mesh_root):
        import fcl
        import numpy as np
        import trimesh

        root = ET.parse(urdf).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            origin, axis, limit, mimic = j.find("origin"), j.find("axis"), j.find("limit"), j.find("mimic")
            self.joints[j.get("name")] = {
                "parent": j.find("parent").get("link"), "child": j.find("child").get("link"), "type": j.get("type"),
                "origin": transform(floats(origin.get("xyz", "0 0 0")), floats(origin.get("rpy", "0 0 0"))),
                "axis": np.array(floats(axis.get("xyz"))) if axis is not None else None,
                "lower": float(limit.get("lower", "-inf")) if limit is not None else None,
                "upper": float(limit.get("upper", "inf")) if limit is not None else None,
                "velocity": float(limit.get("velocity", "inf")) if limit is not None else None,
                "mimic": (mimic.get("joint"), float(mimic.get("multiplier", 1)), float(mimic.get("offset", 0)))
                if mimic is not None else None}
        children = {j["child"] for j in self.joints.values()}
        self.base = next(l.get("name") for l in root.findall("link") if l.get("name") not in children)
        by_parent = {}
        for name, j in self.joints.items():
            by_parent.setdefault(j["parent"], []).append(name)
        self.order, stack = [], [self.base]  # joints, parents before children
        while stack:
            for name in by_parent.get(stack.pop(), []):
                self.order.append(name)
                stack.append(self.joints[name]["child"])

        self.bodies = {}  # link: [(fcl object, link->mesh)]
        for link in root.findall("link"):
            for col in link.findall("collision"):
                mesh = col.find("geometry/mesh")
                if mesh is None:
                    continue
                m = trimesh.load(Path(mesh_root) / mesh.get("filename").replace("package://", ""), force="mesh")
                model = fcl.BVHModel()
                model.beginModel(len(m.vertices), len(m.faces))
                model.addSubModel(np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32))
                model.endModel()
                origin = col.find("origin")
                local = transform(floats(origin.get("xyz", "0 0 0")), floats(origin.get("rpy", "0 0 0"))) \
                    if origin is not None else np.eye(4)
                center = m.bounds.mean(axis=0)  # a cheap bounding sphere, to skip far pairs without touching the meshes
                radius = float(np.linalg.norm(m.vertices - center, axis=1).max())
                self.bodies.setdefault(link.get("name"), []).append(
                    {"obj": fcl.CollisionObject(model), "local": local, "center": center, "radius": radius})
        self.hops = self._hops()

    def _hops(self):
        """Links joined by a chain of at most this many moving joints share a housing; fixed joints do not count."""
        edges = {}
        for j in self.joints.values():
            w = 0 if j["type"] == "fixed" else 1
            edges.setdefault(j["parent"], []).append((j["child"], w))
            edges.setdefault(j["child"], []).append((j["parent"], w))
        out = {}
        for start in edges:
            seen = {start: 0}
            todo = [start]
            while todo:
                a = todo.pop()
                for b, w in edges[a]:
                    if b not in seen or seen[a] + w < seen[b]:
                        seen[b] = seen[a] + w
                        todo.append(b)
            out[start] = seen
        return out

    def positions(self, q):
        """Every joint value: reported, or from its mimic master, or 0 (fingers absent from the file stay open)."""
        out = dict(q)
        for name, j in self.joints.items():
            if name not in out and j["mimic"]:
                master, mult, offset = j["mimic"]
                out[name] = mult * q.get(master, 0.0) + offset
        return out

    def fk(self, q):
        import numpy as np

        q = self.positions(q)
        poses = {self.base: np.eye(4)}
        for name in self.order:
            j = self.joints[name]
            pose = poses[j["parent"]] @ j["origin"]
            if j["type"] in ("revolute", "continuous"):
                pose = pose @ rotation(j["axis"], q.get(name, 0.0))
            poses[j["child"]] = pose
        return poses


def load(path):
    samples = []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            samples.append((s["t"], s["q"]))
    samples.sort(key=lambda s: s[0])
    return samples


def thin(samples):
    """Drop samples where nothing moved, keep the first and the last."""
    kept = samples[:1]
    for s in samples[1:]:
        if max(abs(v - kept[-1][1].get(k, v)) for k, v in s[1].items()) >= MIN_STEP:
            kept.append(s)
    if samples[-1] is not kept[-1]:
        kept.append(samples[-1])
    return kept


def moving_pairs(robot, poses_list, min_hops):
    """Link pairs to check: not neighbours, and their relative pose changes during the motion."""
    import numpy as np

    links = sorted(robot.bodies)
    first = poses_list[0]
    invs = {a: np.linalg.inv(first[a]) for a in links}
    out = []
    for i, a in enumerate(links):
        for b in links[i + 1:]:
            if robot.hops[a].get(b, 99) < min_hops:
                continue
            ref = invs[a] @ first[b]
            if len(poses_list) == 1 or any(np.abs(np.linalg.inv(p[a]) @ p[b] - ref).max() > 1e-4 for p in poses_list):
                out.append((a, b))
    return out


def check_collisions(robot, samples, margin, min_hops):
    """{(a, b): {"first", "last", "hits", "gap"}} over the samples, gap = smallest distance seen, 0 when touching."""
    import fcl
    import numpy as np

    poses_list = [robot.fk(q) for _, q in samples]
    pairs = moving_pairs(robot, poses_list, min_hops)
    report = {}
    for (t, _), poses in zip(samples, poses_list):
        for link, parts in robot.bodies.items():
            for part in parts:
                pose = poses[link] @ part["local"]
                part["obj"].setTransform(fcl.Transform(pose[:3, :3], pose[:3, 3]))
                part["world"] = pose[:3, :3] @ part["center"] + pose[:3, 3]
        for a, b in pairs:
            for pa in robot.bodies[a]:
                for pb in robot.bodies[b]:
                    if np.linalg.norm(pa["world"] - pb["world"]) - pa["radius"] - pb["radius"] > margin:
                        continue
                    oa, ob = pa["obj"], pb["obj"]
                    res = fcl.CollisionResult()
                    hit = fcl.collide(oa, ob, fcl.CollisionRequest(), res) > 0
                    gap = 0.0 if hit else (fcl.distance(oa, ob, fcl.DistanceRequest(), fcl.DistanceResult())
                                           if margin > 0 else float("inf"))
                    r = report.setdefault((a, b), {"first": None, "last": None, "hits": 0, "gap": float("inf"),
                                                   "gap_t": t})
                    if gap < r["gap"]:
                        r["gap"], r["gap_t"] = gap, t
                    if hit:
                        r["first"] = t if r["first"] is None else r["first"]
                        r["last"] = t
                        r["hits"] += 1
    return report, len(pairs)


def check_limits(robot, samples):
    """[text] for joints out of their URDF limits or faster than the URDF velocity."""
    problems = []
    worst = {}
    last = {}
    for t, q in samples:
        for name, v in q.items():
            j = robot.joints.get(name)
            if j is None or j["lower"] is None:
                continue
            over = max(j["lower"] - v, v - j["upper"])
            if over > LIMIT_TOL and over > worst.get(name, (0.0, 0.0))[0]:
                worst[name] = (over, t)
    for name, (over, t) in sorted(worst.items()):
        problems.append(f"limit {name}: {over:.3f} rad outside at t={t:.2f}")
    fastest = {}
    for t, q in samples:
        for name, v in q.items():
            t0, v0 = last.get(name, (None, None))
            if t0 is None:
                last[name] = (t, v)
            elif t - t0 >= SPEED_WINDOW:
                speed = abs(v - v0) / (t - t0)
                j = robot.joints.get(name)
                if j and j["velocity"] and speed > j["velocity"] * SPEED_TOL and speed > fastest.get(name, (0.0,))[0]:
                    fastest[name] = (speed, j["velocity"], t)
                last[name] = (t, v)
    for name, (speed, limit, t) in sorted(fastest.items()):
        problems.append(f"speed {name}: {speed:.2f} rad/s, limit {limit:.2f}, at t={t:.2f}")
    return problems


def check(args):
    samples = load(args.motion)
    if not samples:
        sys.exit(f"{args.motion}: no samples")
    robot = Robot(args.urdf, args.meshes)
    used = thin(samples)
    report, n_pairs = check_collisions(robot, used, args.margin, args.min_hops)
    t0 = samples[0][0]
    print(f"{len(samples)} samples, {len(used)} checked, {samples[-1][0] - t0:.1f} s, {n_pairs} link pairs")

    start_hits, new_hits, near = [], [], []
    for (a, b), r in sorted(report.items(), key=lambda kv: kv[1]["gap"]):
        if r["hits"] and r["first"] == t0:
            start_hits.append(f"{a} - {b}: touching from the start, until t={r['last'] - t0:.2f}, {r['hits']} samples")
        elif r["hits"]:
            new_hits.append(f"{a} - {b}: t={r['first'] - t0:.2f}..{r['last'] - t0:.2f}, {r['hits']} samples")
        elif r["gap"] < args.margin:
            near.append(f"{a} - {b}: gap {r['gap'] * 1000:.0f} mm at t={r['gap_t'] - t0:.2f}")
    limits = check_limits(robot, samples)
    for title, rows in (("Touching at the start", start_hits), ("NEW INTERSECTIONS", new_hits),
                        (f"Closer than {args.margin * 1000:.0f} mm", near), ("LIMITS AND SPEED", limits)):
        print(f"{title}: {len(rows) or 'none'}")
        for row in rows:
            print(f"  {row}")
    sys.exit(1 if new_hits or limits else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("motion")
    chk = sub.add_parser("check")
    chk.add_argument("motion")
    chk.add_argument("--margin", type=float, default=0.02, help="m, report pairs closer than this; 0 is faster")
    chk.add_argument("--min-hops", type=int, default=2,
                     help="skip links this close along the chain: CAD of neighbouring links overlaps at the joint")
    chk.add_argument("--urdf", default=str(URDF))
    chk.add_argument("--meshes", default=str(MESHES))
    args = parser.parse_args()
    if args.cmd == "record":
        record(args.motion)
    else:
        check(args)


if __name__ == "__main__":
    main()
