#!/usr/bin/env python3
"""What is the arm pushing on? Records a run and explains its wrist force peaks. Debug tool, not used in src/ or run.sh.

Record next to tools/move_tcp_check.py (needs ROS, stops on SIGTERM or SIGINT). Writes OUT.joints.jsonl in the format of
tools/trajectory_self_check.py, and OUT.force.jsonl with the wrist wrenches and motor commands:

  python3 tools/motion_trace.py record OUT

Park the objects, to compare the same goals with and without them on the table (debug topic, off in grading):

  python3 tools/motion_trace.py park

Without ROS (numpy, trimesh and the meshes, as trajectory_self_check.py). For every force peak: the force in pelvis,
the lowest mesh point and its gap to the table, trays, arm joints near a limit. Then a time series of that gap:

  python3 tools/motion_trace.py explain OUT [--threshold 20]
  python3 tools/motion_trace.py gaps OUT [--step 1] [--from T] [--to T]

What holds a joint: per arm joint the reading, the command and its speed, the wrench in pelvis, the links nearest the
table and the trays:

  python3 tools/motion_trace.py series OUT right --from T --to T [--step 1]
"""
import argparse
import json
import math
import signal
import time
from pathlib import Path

from trajectory_self_check import MESHES, URDF, Robot

TABLE_Z = -0.18  # top of the table in pelvis, robot.TABLE_SCENE
TILTED = (-0.1792, 0.0700)  # z = a + b x: the table top fitted to depth from the center camera, not measured by touch
TABLE_X = (0.22, 0.98)
TRAYS = {"left": (0.34, 0.69), "right": (0.34, -0.69), "center": (0.53, 0.0)}  # scene.yaml, centre in pelvis
TRAY_SIZE = (0.22, 0.32, 0.07)
HAND_ROOT = {"right": "wrist_roll_r_link", "left": "wrist_roll_l_link"}
ARM_LINKS = ("shoulder", "elbow", "wrist")


def record(out):
    import rclpy
    from bodyctrl_msgs.msg import CmdSetMotorPosition
    from geometry_msgs.msg import WrenchStamped
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("motion_trace", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    joints = open(f"{out}.joints.jsonl", "w")
    force = open(f"{out}.force.jsonl", "w")

    def on_joints(msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        joints.write(json.dumps({"t": t, "q": dict(zip(msg.name, msg.position))}) + "\n")

    def on_wrench(arm, msg):
        f, m = msg.wrench.force, msg.wrench.torque
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        force.write(json.dumps({"t": t, "arm": arm, "f": [f.x, f.y, f.z], "m": [m.x, m.y, m.z]}) + "\n")

    def on_cmd(msg):
        t = node.get_clock().now().nanoseconds * 1e-9
        force.write(json.dumps({"t": t, "cmd": {c.name: [c.pos, c.spd] for c in msg.cmds}}) + "\n")

    node.create_subscription(JointState, "/joint_states", on_joints, qos_profile_sensor_data)
    for arm in ("left", "right"):
        node.create_subscription(WrenchStamped, f"/arm_6dof_{arm}", lambda m, a=arm: on_wrench(a, m), 10)
    node.create_subscription(CmdSetMotorPosition, "/arm/cmd_pos", on_cmd, 10)
    while rclpy.ok() and not stop:
        rclpy.spin_once(node, timeout_sec=0.1)
    joints.close()
    force.close()
    node.destroy_node()
    rclpy.try_shutdown()


def park():
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("motion_trace_park")
    names = []
    node.create_subscription(String, "/sim/objects/names", lambda m: names.append(json.loads(m.data)),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    pub = node.create_publisher(String, "/sim/objects/set", 10)
    t0 = time.time()
    while not names and time.time() - t0 < 30.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not names:
        raise SystemExit("no /sim/objects/names")
    msg = String(data=json.dumps({name: "park" for name in names[0]}))
    while pub.get_subscription_count() == 0 and time.time() - t0 < 30.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    pub.publish(msg)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"parked {len(names[0])} objects: {sorted(names[0])}")
    node.destroy_node()
    rclpy.try_shutdown()


def load_force(out):
    wrenches, cmds = {"left": [], "right": []}, []
    with open(f"{out}.force.jsonl") as f:
        for line in f:
            s = json.loads(line)
            if "arm" in s:
                wrenches[s["arm"]].append(s)
            elif "cmd" in s:
                cmds.append(s)
    return wrenches, cmds


def vertices(robot):
    """{link: [N x 3 mesh vertices in the link frame, thinned]} for the links of the arms and hands."""
    import numpy as np
    import trimesh
    import xml.etree.ElementTree as ET

    root = ET.parse(URDF).getroot()
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            m = trimesh.load(Path(MESHES) / mesh.get("filename").replace("package://", ""), force="mesh")
            v = np.asarray(m.vertices)[::max(1, len(m.vertices) // 4000)]
            local = robot.bodies[name][len(out.get(name, []))]["local"]
            out.setdefault(name, []).append(v @ local[:3, :3].T + local[:3, 3])
    return {k: np.vstack(v) for k, v in out.items()}


def side_links(robot, side):
    """Links below the shoulder of an arm: the shoulder_pitch joint's subtree."""
    s = side[0]
    start = f"shoulder_pitch_{s}_joint"
    out = set()
    todo = [start]
    while todo:
        j = robot.joints[todo.pop()]
        out.add(j["child"])
        todo += [n for n, o in robot.joints.items() if o["parent"] == j["child"]]
    return out


def clearance(robot, verts, q, side):
    """The mesh point of the arm closest to the tilted table top, and the links inside a tray."""
    import numpy as np

    poses = robot.fk(q)
    # pelvis is not the URDF base when these differ; the sample's joints do not move the pelvis, so re-anchor there
    inv = np.linalg.inv(poses["pelvis"]) if "pelvis" in poses else np.eye(4)
    low = (math.inf, None, None, None)
    trays = []
    for link in side_links(robot, side):
        if link not in verts:
            continue
        pose = inv @ poses[link]
        world = verts[link] @ pose[:3, :3].T + pose[:3, 3]
        over = (world[:, 0] > TABLE_X[0]) & (np.abs(world[:, 1]) < 1.22)
        if over.any():
            gap = world[over, 2] - (TILTED[0] + TILTED[1] * world[over, 0])
            if gap.min() < low[0]:
                i = np.flatnonzero(over)[gap.argmin()]
                low = (float(gap.min()), link, world[i].round(3).tolist(), float(world[over, 2].min() - TABLE_Z))
        for name, (cx, cy) in TRAYS.items():
            h = world[:, 2] < TABLE_Z + TRAY_SIZE[2]
            inside = h & (np.abs(world[:, 0] - cx) < TRAY_SIZE[0] / 2) & (np.abs(world[:, 1] - cy) < TRAY_SIZE[1] / 2)
            if inside.any():
                trays.append(f"{link} in tray_{name}")
    return low, trays, poses, inv


def at(rows, t):
    """The last row at or before t."""
    best = None
    for row in rows:
        if row["t"] > t:
            break
        best = row
    return best


def explain(out, threshold):
    import numpy as np
    from trajectory_self_check import load

    robot = Robot(URDF, MESHES)
    verts = vertices(robot)
    samples = load(f"{out}.joints.jsonl")
    joints = [{"t": t, "q": q} for t, q in samples]
    wrenches, cmds = load_force(out)
    t0 = samples[0][0]
    print(f"{len(samples)} joint samples, {samples[-1][0] - t0:.1f} s; command rows {len(cmds)}")
    for side, rows in wrenches.items():
        rows.sort(key=lambda r: r["t"])
        if not rows:
            continue
        norm = np.array([np.linalg.norm(r["f"]) for r in rows])
        print(f"{side}: {len(rows)} wrench rows, peak {norm.max():.0f} N, above {threshold} N for "
              f"{(norm > threshold).sum()} rows")
        # one line per contiguous episode above the threshold
        i = 0
        while i < len(rows):
            if norm[i] <= threshold:
                i += 1
                continue
            j = i
            while j + 1 < len(rows) and (norm[j + 1] > threshold or rows[j + 1]["t"] - rows[j]["t"] < 0.2 and norm[j + 1] > threshold / 2):
                j += 1
            k = i + int(norm[i:j + 1].argmax())
            row = rows[k]
            q = at(joints, row["t"])["q"]
            low, trays, poses, inv = clearance(robot, verts, q, side)
            pose = inv @ poses[HAND_ROOT[side]]
            force = pose[:3, :3] @ np.array(row["f"])
            names = [n for n in q if f"_{side[0]}_joint" in n and n.split("_")[0] in ARM_LINKS]
            near = []
            for n in names:
                joint = robot.joints[n]
                gap = min(q[n] - joint["lower"], joint["upper"] - q[n])
                if gap < 0.05:
                    near.append(f"{n} {q[n]:+.3f} ({gap:+.3f} to limit)")
            d = [q[n] - at(joints, row["t"] - 0.2)["q"][n] if at(joints, row["t"] - 0.2) else 0.0 for n in names]
            print(f"  t={row['t'] - t0:7.2f}..{rows[j]['t'] - t0:7.2f} s  peak {norm[k]:5.0f} N  "
                  f"F pelvis ({force[0]:+5.0f},{force[1]:+5.0f},{force[2]:+5.0f})  "
                  f"lowest {low[1]} at {low[2]} gap tilted {low[0]:+.3f} flat {low[3]:+.3f}  "
                  f"tcp {(inv @ poses[f'{side}_tcp_link'])[:3, 3].round(3).tolist()}  "
                  f"speed {max(abs(v) for v in d) / 0.2:.2f} rad/s  "
                  f"{'; '.join(trays) or 'no tray'}  limits: {'; '.join(near) or 'none'}", flush=True)
            i = j + 1


def bent(q, side):
    """The finger joint furthest from its command: the fingers stay open here, so a reading is something pushing them."""
    names = [n for n in q if n.startswith(f"{side}_") and n.split("_")[1] in ("thumb", "index", "middle", "ring", "little")]
    name = max(names, key=lambda n: abs(q[n]))
    return f"{name.replace(side + '_', '').replace('_joint', '')} {q[name]:+.3f}"


def fingers(out):
    """How the open hand meets the table: per finger, the height of the lowest point under the TCP over the table top.

    top_down is level in pelvis, so 4 degrees off the table; the same hand along the table normal is printed next to it.
    The geometry comes from one sample of a recording with open fingers, moved to the goal orientation.
    """
    import numpy as np
    from trajectory_self_check import load

    robot = Robot(URDF, MESHES)
    verts = vertices(robot)
    _, q = load(f"{out}.joints.jsonl")[0]
    tilt = math.atan(TILTED[1])
    normal = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])
    level = np.array([[math.cos(tilt), 0.0, -math.sin(tilt)], [0.0, 1.0, 0.0], [math.sin(tilt), 0.0, math.cos(tilt)]])
    open_q = {n: (0.0 if n.split("_")[1] in ("thumb", "index", "middle", "ring", "little") else v) for n, v in q.items()}
    poses = robot.fk(open_q)
    for side in ("left", "right"):
        inv = np.linalg.inv(poses[f"{side}_tcp_link"])
        tips = {}
        for link in side_links(robot, side):
            part = link.split("_")[1] if link.startswith(side) else ""
            if link in verts and part in ("thumb", "index", "middle", "ring", "little"):
                pose = inv @ poses[link]
                tips.setdefault(part, []).append(verts[link] @ pose[:3, :3].T + pose[:3, 3])
        tips = {k: np.vstack(v) for k, v in tips.items()}
        for yaw in (0.0, 0.8, math.pi / 2):
            spin = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
            rows = []
            for part, v in sorted(tips.items()):
                a = (v @ spin.T) @ normal
                b = (v @ spin.T @ level.T) @ normal
                rows.append(f"{part} {-a.min():.4f}/{-b.min():.4f}")
            print(f"{side} yaw {yaw:.2f} under the TCP along the normal, top_down/aligned, m: " + " ".join(rows), flush=True)


def gaps(out, step, start, end):
    """Time series: per arm, TCP, the lowest mesh point against the tilted and the flat table, wrist force and torque."""
    import numpy as np
    from trajectory_self_check import load

    robot = Robot(URDF, MESHES)
    verts = vertices(robot)
    samples = load(f"{out}.joints.jsonl")
    wrenches, _ = load_force(out)
    t0 = samples[0][0]
    for rows in wrenches.values():
        rows.sort(key=lambda r: r["t"])
    print("   t  arm  tcp x, y, z          lowest link          at x, y, z            tilted    flat    |F|   |M|  finger")
    next_t = start
    for t, q in samples:
        if t - t0 < next_t:
            continue
        if t - t0 > end:
            break
        next_t = t - t0 + step
        for side in ("left", "right"):
            low, _, poses, inv = clearance(robot, verts, q, side)
            row = at(wrenches[side], t)
            if low[1] is None or row is None:
                continue
            tcp = (inv @ poses[f"{side}_tcp_link"])[:3, 3]
            print(f"{t - t0:6.2f} {side[0]}  {tcp[0]:+.3f} {tcp[1]:+.3f} {tcp[2]:+.3f}  {low[1]:<20} "
                  f"{low[2][0]:+.3f} {low[2][1]:+.3f} {low[2][2]:+.3f}  {low[0]:+.3f}  {low[3]:+.3f}  "
                  f"{np.linalg.norm(row['f']):5.1f} {np.linalg.norm(row['m']):5.2f}  {bent(q, side)}", flush=True)


def series(out, side, step, start, end, links=()):
    import numpy as np
    from trajectory_self_check import load

    robot = Robot(URDF, MESHES)
    verts = vertices(robot)
    samples = load(f"{out}.joints.jsonl")
    wrenches, cmds = load_force(out)
    t0 = samples[0][0]
    rows = sorted(wrenches[side], key=lambda r: r["t"])
    s = side[0]
    arm = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll"]
    base = 21 if side == "right" else 11
    next_t = start
    for t, q in samples:
        if t - t0 < next_t:
            continue
        if t - t0 > end:
            break
        next_t = t - t0 + step
        poses = robot.fk(q)
        inv = np.linalg.inv(poses["pelvis"])
        wrench = at(rows, t)
        rot = (inv @ poses[HAND_ROOT[side]])[:3, :3]
        tcp = (inv @ poses[f"{side}_tcp_link"])[:3, 3]
        print(f"t {t - t0:6.2f} tcp {tcp.round(3).tolist()} F pelvis {(rot @ np.array(wrench['f'])).round(1).tolist()} "
              f"M pelvis {(rot @ np.array(wrench['m'])).round(2).tolist()}")
        latest = {}
        for c in cmds:
            if c["t"] > t:
                break
            latest.update(c["cmd"])
        print("   " + "  ".join(f"{n} {q[f'{n}_{s}_joint']:+.3f}/{latest.get(str(base + i), [math.nan])[0]:+.3f}"
                                f"@{latest.get(str(base + i), [0, math.nan])[1]:.2f}" for i, n in enumerate(arm)))
        near = []
        for link in side_links(robot, side):
            if link not in verts:
                continue
            pose = inv @ poses[link]
            world = verts[link] @ pose[:3, :3].T + pose[:3, 3]
            over = world[:, 0] > TABLE_X[0]
            table = (world[over, 2] - (TILTED[0] + TILTED[1] * world[over, 0])).min() if over.any() else math.inf
            tray = math.inf
            for cx, cy in TRAYS.values():
                d = np.maximum(np.abs(world[:, :2] - (cx, cy)) - np.array(TRAY_SIZE[:2]) / 2, 0.0)
                d = np.linalg.norm(d, axis=1)
                low = world[:, 2] < TILTED[0] + TILTED[1] * world[:, 0] + TRAY_SIZE[2]
                if low.any():
                    tray = min(tray, float(d[low].min()))
            near.append((min(table, tray), link, table, tray))
        print("   " + "  ".join(f"{link} table {table:+.3f} tray {tray:.3f}" for _, link, table, tray in sorted(near)[:5]),
              flush=True)
        fingers = sorted(n for n in q if side in n and any(f in n for f in ("thumb", "index")))
        print("   " + " ".join(f"{n.replace(side + '_', '').replace('_joint', '')} {q[n]:+.3f}" for n in fingers))
        for link in links:
            pose = inv @ poses[link]
            world = verts[link] @ pose[:3, :3].T + pose[:3, 3]
            print(f"   {link} box x {world[:, 0].min():+.3f}..{world[:, 0].max():+.3f} "
                  f"y {world[:, 1].min():+.3f}..{world[:, 1].max():+.3f} z {world[:, 2].min():+.3f}..{world[:, 2].max():+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("out")
    sub.add_parser("park")
    exp = sub.add_parser("explain")
    exp.add_argument("out")
    exp.add_argument("--threshold", type=float, default=20.0)
    gap = sub.add_parser("gaps")
    gap.add_argument("out")
    gap.add_argument("--step", type=float, default=1.0)
    gap.add_argument("--from", dest="start", type=float, default=0.0)
    gap.add_argument("--to", dest="end", type=float, default=math.inf)
    sub.add_parser("fingers").add_argument("out")
    ser = sub.add_parser("series")
    ser.add_argument("out")
    ser.add_argument("side", choices=("left", "right"))
    ser.add_argument("--step", type=float, default=1.0)
    ser.add_argument("--from", dest="start", type=float, default=0.0)
    ser.add_argument("--to", dest="end", type=float, default=math.inf)
    ser.add_argument("--links", default="", help="comma separated: print the bounding box of these links too")
    args = parser.parse_args()
    if args.cmd == "record":
        record(args.out)
    elif args.cmd == "park":
        park()
    elif args.cmd == "series":
        series(args.out, args.side, args.step, args.start, args.end, [x for x in args.links.split(",") if x])
    elif args.cmd == "fingers":
        fingers(args.out)
    elif args.cmd == "gaps":
        gaps(args.out, args.step, args.start, args.end)
    else:
        explain(args.out, args.threshold)


if __name__ == "__main__":
    main()
