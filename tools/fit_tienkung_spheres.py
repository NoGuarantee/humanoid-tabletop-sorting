#!/usr/bin/env python3
"""Offline: build the cuRobo robot config for TienKung 2.0 Pro (collision spheres + self-collision matrix).

The simulator URDF references STL meshes that are not shipped anywhere we may use. The same URDF (identical except
comments) and its meshes are published by the manufacturer in Open-X-Humanoid/TienKung_URDF, package
tiangong2pro_urdf, under the OpenAtom Open Hardware License 1.0. This script downloads them at a pinned commit,
fits spheres with cuRobo's RobotBuilder and writes the config into src/hack_motion/robot/. Only the URDF, the
config and the licence go into git; the meshes stay in data/ (git-ignored, 120 MB) and are not needed at runtime.

On the server, in a fresh container from the solution image (cuRobo is installed first):

  bash tools/remote.sh dev --timeout 3600 -- \\
    'bash tools/curobo_install.sh && python3 tools/fit_tienkung_spheres.py'
  bash tools/remote.sh fetch work/$NSSR_WHO/nssr-sol-team-049/src/hack_motion/robot .fetch  # before any sync
  cp .fetch/robot/tienkung_curobo.yml src/hack_motion/robot/

Steps (default: all): fetch (URDF, licence, meshes), fit (spheres, collision matrix), check (load the config
without meshes the way motion does at runtime, FK/IK smoke test).
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO = "Open-X-Humanoid/TienKung_URDF"
SHA = "4c2c061d72dcdab60d74e45d3efb17246a3b6f62"  # 2026-01-27, matches the simulator URDF
PKG = "tiangong2pro_urdf"
SRC_URDF = f"{PKG}/urdf/tiangong2.0_pro_with_hands.urdf"

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "tienkung_urdf" / SHA[:12]  # contains tiangong2pro_urdf/{urdf,meshes}
OUT_DIR = ROOT / "src" / "hack_motion" / "robot"
OUT_URDF = OUT_DIR / "tiangong2pro_with_hands.urdf"
OUT_CFG = OUT_DIR / "tienkung_curobo.yml"
OUT_LICENCE = OUT_DIR / "LICENSE.TienKung_URDF"

TOOL_FRAMES = ["left_tcp_link", "right_tcp_link"]

# Links that get spheres, and the density multiplier for each group. Feet are far below anything the arms reach;
# imu_link sits inside the pelvis.
SKIP_LINKS = {"ankle_pitch_l_link", "ankle_roll_l_link", "ankle_pitch_r_link", "ankle_roll_r_link", "imu_link"}
LEG_LINKS = {f"{j}_{s}_link" for j in ("hip_roll", "hip_pitch", "hip_yaw", "knee_pitch") for s in "lr"}
LEG_DENSITY = 0.5

# Fingers are not in the cuRobo tree: their spheres, at the open pose (all finger joints 0), are moved onto the palm
# link. The hand is driven by the hand presets, not by the planner, and cuRobo only supports joint axes along x/y/z,
# while right_index_1_joint in the published URDF is tilted by 4 degrees. At q = 0 the axis does not matter.
PALMS = {"left": "L_base_link", "right": "R_base_link"}

# Home pose of the robot in the simulator (TIANGONG_PRO_HOME_POSE in its tiangong_pro/config.py): knees bent,
# arms down.
# Joints not listed are 0. Values of the locked joints.
HOME_POSE = {**{f"hip_pitch_{s}_joint": -0.5 for s in "lr"}, **{f"knee_pitch_{s}_joint": 1.0 for s in "lr"}}
# cspace.default_joint_position: home with the arms 0.2 rad away from the body. The builder ignores link pairs
# that touch in the default pose; at home the forearm spheres touch the thighs by a few mm, and ignoring that pair
# would let the planner put a forearm through a thigh. Motion moves to this pose by joints before planning.
READY_POSE = {**HOME_POSE, "shoulder_roll_l_joint": 0.2, "shoulder_roll_r_joint": -0.2}
# A palm-down hand folds the wrist: wrist_roll comes within a fraction of a millimetre of elbow_yaw, two joints
# away. The ready pose never brings them together and the sampled poses rarely do, so the builder keeps the pair
# and every top-down hand is then "in collision". The two links are joined through wrist_pitch and cannot pass
# through each other, so the pair only gives false positives.
FOLDED_WRIST = {f"elbow_yaw_{s}_link": [f"wrist_roll_{s}_link"] for s in "lr"}
# Pairs the builder must keep checking, whatever the sampled poses say: the forearm and hand against the pelvis and
# legs. compute_collision_matrix drops every pair that never touched in its random samples, and it dropped
# elbow_pitch against pelvis and hip_roll, so a hanging arm could be planned through the hip.
FOREARM = {f"{j}_{s}_link" for j in ("elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll") for s in "lr"} | {
    "L_base_link", "R_base_link"}
LOWER_BODY = {"pelvis"} | {f"{j}_{s}_link" for j in ("hip_roll", "hip_pitch", "hip_yaw", "knee_pitch") for s in "lr"}

# Joints that cuRobo keeps fixed. Legs and head: motion overrides them from /joint_states before creating the
# solver. Ankle joints are not in the cuRobo tree at all (feet have no spheres), so they are not listed.
LOCK_JOINTS = {j: HOME_POSE.get(j, 0.0) for j in (
    *(f"{j}_{s}_joint" for j in ("hip_roll", "hip_pitch", "hip_yaw", "knee_pitch") for s in "lr"),
    "head_yaw_joint", "head_pitch_joint", "head_roll_joint")}



def rigid_pairs(urdf_path, links):
    """{link: [links]} for every pair of `links` joined only by fixed and LOCK_JOINTS joints: planning never moves
    them relative to each other, so checking them is useless. Motion re-locks them from /joint_states, and a pair
    that clears at the generator's lock can touch at the real one: at any head_pitch other than 0, head_roll
    touches waist_yaw and every IK goal comes out infeasible."""
    import xml.etree.ElementTree as ET
    parent = {}
    for j in ET.parse(urdf_path).getroot().iter("joint"):
        a, b = j.find("parent").get("link"), j.find("child").get("link")
        parent[b] = a if j.get("type") == "fixed" or j.get("name") in LOCK_JOINTS else None

    def root(link):
        while parent.get(link):
            link = parent[link]
        return link

    groups = {}
    for link in links:
        groups.setdefault(root(link), []).append(link)
    return {g[i]: g[i + 1:] for g in groups.values() for i in range(len(g) - 1)}


# For the report: self-collision ignores between these groups are suspicious (the arm could then pass through
# the body unnoticed), so they are printed for a human to look at.
def link_group(name):
    if name in LEG_LINKS or name.startswith(("ankle_", "hip_", "knee_")):
        return "legs"
    for side, tag, hand in (("left", "_l_", "L_"), ("right", "_r_", "R_")):
        if tag in name + "_" or name.startswith((side + "_", hand)):
            return side
    return "body"  # pelvis, waist, head, cameras


def log(*a):
    print("[fit_tienkung]", *a, flush=True)


# ---------------------------------------------------------------- fetch

def http_get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nssr-team049-tools"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccups on the server
            if i == tries - 1:
                raise
            log(f"retry {url}: {e}")
            time.sleep(2 * (i + 1))


def git_blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch():
    tree = json.loads(http_get(f"https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1"))
    if tree.get("truncated"):
        sys.exit("tree listing truncated")
    blobs = {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}
    want = [p for p in blobs if p.startswith(f"{PKG}/meshes/")] + [SRC_URDF, "licence"]
    done = 0
    for path in want:
        dst = CACHE / path
        if dst.exists() and git_blob_sha(dst.read_bytes()) == blobs[path]:
            continue
        data = http_get(f"https://raw.githubusercontent.com/{REPO}/{SHA}/{path}")
        if git_blob_sha(data) != blobs[path]:
            sys.exit(f"checksum mismatch: {path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        done += 1
    log(f"{len(want)} files in {CACHE} ({done} downloaded), all match git blob hashes at {SHA[:12]}")

    # The URDF and the licence go into the package. The URDF is the published one byte for byte.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CACHE / SRC_URDF, OUT_URDF)
    shutil.copyfile(CACHE / "licence", OUT_LICENCE)
    log(f"wrote {OUT_URDF.relative_to(ROOT)} and {OUT_LICENCE.relative_to(ROOT)}")


# ---------------------------------------------------------------- fit

def urdf_zero_pose_transforms(urdf_path):
    """{child_link: (parent_link, 4x4 parent_T_child at joint value 0)} from the URDF joint origins."""
    import xml.etree.ElementTree as ET
    import numpy as np

    def rot(r, p, y):
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return rz @ ry @ rx  # URDF fixed-axis roll, pitch, yaw

    out = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        t = np.eye(4)
        t[:3, :3] = rot(*rpy)
        t[:3, 3] = xyz
        out[j.find("child").get("link")] = (j.find("parent").get("link"), t)
    return out


def merge_fingers_into_palms(spheres, urdf_path):
    """Move finger spheres onto the palm links at the open pose; returns the new dict and the moved links."""
    import numpy as np
    tf = urdf_zero_pose_transforms(urdf_path)
    moved = []
    for side, palm in PALMS.items():
        for link in [n for n in spheres if n.startswith(side + "_") and n != f"{side}_tcp_link"]:
            t, cur = np.eye(4), link
            while cur != palm:  # palm_T_link = product of joint origins up the tree
                parent, pt = tf[cur]
                t, cur = pt @ t, parent
            for s in spheres.pop(link):
                c = t @ np.array([*s["center"], 1.0])
                spheres[palm].append({"center": c[:3].tolist(), "radius": s["radius"]})
            moved.append(link)
    return spheres, moved


def colliding_pairs(kin_cfg, position):
    """[(link_a, link_b, depth_m)] for sphere pairs in collision at one joint position (same check as the builder)."""
    import torch
    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
    from curobo._src.robot.kinematics.kinematics import Kinematics

    kin = Kinematics(kin_cfg)
    js = kin.default_joint_state.clone()
    if position is not None:
        js.position = position.to(js.position)
    js.position = js.position.view(1, -1)
    cost = SelfCollisionCost(SelfCollisionCostCfg(weight=1.0, store_pair_distance=True,
                                                  self_collision_kin_config=kin_cfg.self_collision_config))
    cost.setup_batch_tensors(1, 1)
    cost.forward(kin.compute_kinematics(js).robot_spheres)
    dist = cost._pair_distance.view(-1)
    hit = dist > 0
    if not torch.any(hit):
        return []
    sph = kin_cfg.self_collision_config.collision_pairs[hit].to(torch.int64)
    link_idx = kin_cfg.kinematics_config.link_sphere_idx_map.to(torch.int64)[sph].cpu().tolist()
    names = {v: k for k, v in kin_cfg.kinematics_config.link_name_to_idx_map.items()}
    worst = {}
    for (a, b), d in zip(link_idx, dist[hit].cpu().tolist()):
        key = tuple(sorted((names[a], names[b])))
        worst[key] = max(worst.get(key, 0.0), d)
    return sorted(((a, b, d) for (a, b), d in worst.items()), key=lambda x: -x[2])


def use_home_pose_in_builder(builder):
    """RobotBuilder builds its temporary config without cspace, so "collides at the default configuration" means
    the middle of every joint range (arms half raised, elbows bent). Give it READY_POSE instead."""
    import types
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg

    if builder._self_collision_ignore is None:  # neighbours only; compute_collision_matrix starts from these too
        builder._self_collision_ignore = builder._create_neighbor_ignore_matrix()
    names = builder._build_temp_config().cspace.joint_names
    cspace = {"joint_names": list(names), "default_joint_position": [READY_POSE.get(j, 0.0) for j in names]}

    def build_temp_config(self):  # RobotBuilder._build_temp_config of v0.8.0 plus cspace
        cfg = KinematicsLoaderCfg(
            base_link=self._base_link, tool_frames=self._tool_frames, urdf_path=self.urdf_path,
            asset_root_path=self.asset_path, collision_link_names=list(self._collision_spheres.keys()),
            collision_spheres=self._collision_spheres, mesh_link_names=list(self._collision_spheres.keys()),
            self_collision_buffer=self._self_collision_buffer, self_collision_ignore=self._self_collision_ignore,
            device_cfg=self.device_cfg, load_meshes=False, cspace=dict(cspace))
        return KinematicsCfg.from_data_dict(vars(cfg))

    builder._build_temp_config = types.MethodType(build_temp_config, builder)
    return cspace


def fit(args):
    import torch
    import yaml
    from curobo.robot_builder import RobotBuilder

    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    builder = RobotBuilder(urdf_path=str(CACHE / SRC_URDF), asset_path=str(CACHE), tool_frames=TOOL_FRAMES)
    mesh_links = list(builder._mesh_link_names)  # links with geometry, in URDF order
    links = [n for n in mesh_links if n not in SKIP_LINKS]
    log(f"{len(mesh_links)} links with meshes, fitting {len(links)}; skipped: {sorted(SKIP_LINKS & set(mesh_links))}")

    t = time.perf_counter()
    for name in links:
        density = LEG_DENSITY if name in LEG_LINKS else args.density
        builder.refit_link_spheres(name, sphere_density=density, compute_metrics=True)
    log(f"fitted {builder.num_spheres} spheres in {time.perf_counter() - t:.0f} s")

    print(f"\n  {'link':<24s} {'n':>3s} {'cover%':>7s} {'protr%':>7s} {'prot_mm':>8s} {'gap_mm':>7s}")
    for name, m in builder.link_metrics.items():
        print(f"  {name:<24s} {m.num_spheres:3d} {m.coverage * 100:6.1f}% {m.protrusion * 100:6.1f}% "
              f"{m.protrusion_dist_mean * 1e3:7.2f} {m.surface_gap_mean * 1e3:6.2f}")

    # RobotBuilder has no public setter for this; the dict is what fit/refit fill and build() reads.
    builder._collision_spheres, moved = merge_fingers_into_palms(builder._collision_spheres, str(CACHE / SRC_URDF))
    log(f"moved spheres of {len(moved)} finger links onto {list(PALMS.values())}; "
        f"{len(builder.collision_link_names)} links with spheres")

    use_home_pose_in_builder(builder)
    home = colliding_pairs(builder._build_temp_config(), None)
    log(f"{len(home)} link pairs touch in the ready pose; the builder ignores them, cross-group ones are suspicious:")
    for a, b, d in home:
        flag = "  <-- cross-group" if link_group(a) != link_group(b) else ""
        print(f"    {a:<24s} {b:<24s} {d * 1e3:6.1f} mm{flag}")

    t = time.perf_counter()
    rigid = rigid_pairs(str(OUT_URDF), builder.collision_link_names)
    log(f"{sum(map(len, rigid.values()))} pairs of links joined only by fixed and locked joints, ignored")
    custom = {k: list(v) for k, v in rigid.items()}
    for a, bs in FOLDED_WRIST.items():
        custom.setdefault(a, []).extend(bs)
    ignore = builder.compute_collision_matrix(num_samples=args.collision_samples, seed=args.seed,
                                              custom_ignore=custom)
    log(f"collision matrix in {time.perf_counter() - t:.0f} s")
    kept = []
    for a, bs in ignore.items():
        for b in list(bs):
            if (a in FOREARM and b in LOWER_BODY) or (a in LOWER_BODY and b in FOREARM):
                bs.remove(b)
                kept.append((a, b))
    log(f"kept checking {len(kept)} forearm/hand vs pelvis/leg pairs the sampling would ignore: {kept}")
    builder._self_collision_ignore = ignore  # build() writes this one
    cross = sorted({tuple(sorted((a, b))) for a, bs in ignore.items() for b in bs
                    if link_group(a) != link_group(b)})
    log(f"{len(cross)} ignored pairs across groups (left/right/body/legs); check these are really harmless:")
    for a, b in cross:
        print(f"    {a:<24s} {b}")

    config = builder.build()
    tmp = OUT_CFG.with_suffix(".tmp.yml")
    builder.save(config, str(tmp))
    data = yaml.safe_load(tmp.read_text())
    tmp.unlink()
    k = data["kinematics"]
    # Paths: the config is loaded by motion, which sets urdf_path to the installed copy. No meshes at runtime.
    k["urdf_path"] = OUT_URDF.name
    k["asset_root_path"] = ""
    k["load_meshes"] = False
    k["lock_joints"] = LOCK_JOINTS
    cs = k["cspace"]
    cs["default_joint_position"] = [READY_POSE.get(j, 0.0) for j in cs["joint_names"]]
    for key in ("load_collision_spheres", "num_envs"):  # solver arguments, passed again by *Cfg.create
        k.pop(key, None)
    header = (f"# TienKung 2.0 Pro for cuRobo v0.8, generated by tools/fit_tienkung_spheres.py, do not edit by hand.\n"
              f"# Collision spheres fitted to the meshes of {REPO}@{SHA[:12]}/{PKG}\n"
              f"# (OpenAtom Open Hardware License 1.0, see LICENSE.TienKung_URDF): derived data, not the original\n"
              f"# files. Finger spheres are on the palm links, at the open pose.\n"
              f"# Tool frames: {', '.join(TOOL_FRAMES)}. lock_joints: the simulator home pose (motion replaces them with\n"
              f"# /joint_states). Default pose: home with the arms 0.2 rad away from the body (shoulder_roll).\n")
    OUT_CFG.write_text(header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=120))
    log(f"wrote {OUT_CFG.relative_to(ROOT)}: {builder.num_spheres} spheres on {len(builder.collision_link_names)} "
        f"links, {sum(len(v) for v in ignore.values())} ignore entries")


# ---------------------------------------------------------------- check

def load_runtime_cfg():
    """What motion will do: read the yml, point urdf_path at the package copy, no mesh directory."""
    import yaml
    data = yaml.safe_load(OUT_CFG.read_text())
    data["kinematics"]["urdf_path"] = str(OUT_URDF)
    data["kinematics"]["asset_root_path"] = ""
    return data


HEAD_EXTREMES = [{"head_yaw_joint": y, "head_pitch_joint": p, "head_roll_joint": r}
                 for y in (-1.5, 0.0, 1.5) for p in (-0.43, 0.43) for r in (-0.45, 0.45)]


def check(args):
    import torch
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import JointState

    hidden = CACHE.with_name(CACHE.name + ".hidden")
    if CACHE.exists():  # prove the meshes are not needed
        CACHE.rename(hidden)
    try:
        t = time.perf_counter()
        ik = InverseKinematics(InverseKinematicsCfg.create(robot=load_runtime_cfg(), num_seeds=32,
                                                           self_collision_check=True))
        log(f"IK created in {time.perf_counter() - t:.1f} s; {len(ik.joint_names)} active joints: {ik.joint_names}")
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        kin_cfg = KinematicsCfg.from_data_dict(load_runtime_cfg()["kinematics"])
        for name, pose in (("ready (default)", READY_POSE), ("home (arms down)", HOME_POSE)):
            pairs = colliding_pairs(kin_cfg, torch.tensor([pose.get(j, 0.0) for j in ik.joint_names]))
            log(f"{name}: {len(pairs)} colliding link pairs" + "".join(
                f"\n    {a:<24s} {b:<24s} {d * 1e3:6.1f} mm" for a, b, d in pairs))
            if name.startswith("ready") and pairs:
                sys.exit("the default pose is in collision")
        for lock in HEAD_EXTREMES:  # motion re-locks the head from /joint_states: ready must stay clear
            data = load_runtime_cfg()
            data["kinematics"]["lock_joints"].update(lock)
            pairs = colliding_pairs(KinematicsCfg.from_data_dict(data["kinematics"]),
                                    torch.tensor([READY_POSE.get(j, 0.0) for j in ik.joint_names]))
            if pairs:
                sys.exit(f"ready with the head at {lock} is in collision: {pairs}")
        log(f"ready is clear at {len(HEAD_EXTREMES)} head extremes")
        q = ik.sample_configs(args.n)  # collision-free, so FK of each is a reachable goal for both hands
        log(f"sample_configs: {q.shape[0]}/{args.n} collision-free configurations")
        if q.shape[0] == 0:
            sys.exit("no collision-free configurations: something collides everywhere, see the pairs above")
        args.n = q.shape[0]
        ok, errs, times = 0, [], []
        for i in range(args.n):
            js = JointState.from_position(q[i:i + 1], joint_names=ik.joint_names)
            g = ik.compute_kinematics(js).tool_poses.as_goal()
            t = time.perf_counter()
            r = ik.solve_pose(g)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            ok += int(r.success.item())
            errs.append(r.position_error.max().item())
        times.sort()
        log(f"IK both hands, {args.n} reachable goals: success {ok}/{args.n}, "
            f"max pos err {max(errs) * 1e3:.2f} mm, median {times[len(times) // 2] * 1e3:.0f} ms")
    finally:
        if hidden.exists():
            hidden.rename(CACHE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="fetch, fit, check (default: all)")
    ap.add_argument("--density", type=float, default=1.0, help="sphere density for arms, hands, torso, head")
    ap.add_argument("--collision-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=49)
    ap.add_argument("--n", type=int, default=20, help="IK goals in check")
    args = ap.parse_args()
    args.steps = args.steps or ["fetch", "fit", "check"]  # Python 3.10 checks a list default against choices
    if set(args.steps) - {"fetch", "fit", "check"}:
        ap.error(f"unknown step in {args.steps}")
    if "fetch" in args.steps:
        fetch()
    if "fit" in args.steps:
        fit(args)
    if "check" in args.steps:
        check(args)


if __name__ == "__main__":
    main()
