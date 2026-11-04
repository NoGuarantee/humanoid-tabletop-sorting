#!/usr/bin/env python3
"""Runs the table_clusters module over captured frames, without ROS, and prints what it found and how long it took.

    python3 tools/perception_replay.py FRAMES_DIR OUT_DIR [--poses center,left,right] [--cams head,torso]

FRAMES_DIR/layoutN/<pose>_cam_<camera>.npz hold rgb, depth (uint16, mm, 0 = none), K, T_pelvis_camera.
Pictures with the clusters drawn over rgb go to OUT_DIR.
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "table_clusters", os.path.join(HERE, "..", "src", "hack_perception", "hack_perception", "table_clusters.py"))
tc = importlib.util.module_from_spec(spec)
sys.modules["table_clusters"] = tc
spec.loader.exec_module(tc)


def cloud_from_frame(d):
    z = d["depth"].astype(np.float64) * 1e-3
    K, T = d["K"], d["T_pelvis_camera"]
    v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)
    cloud = p @ T[:3, :3].T + T[:3, 3]
    cloud[z == 0] = np.nan
    return cloud


PALETTE = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255),
           (128, 0, 255), (0, 255, 128), (255, 128, 0)]


def draw(rgb, result, title):
    img = rgb[..., ::-1].copy()
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3)
    for k, c in enumerate(result.objects):
        color = np.array(PALETTE[k % len(PALETTE)], dtype=np.uint8)
        flat[c.pixels] = (0.45 * flat[c.pixels] + 0.55 * color).astype(np.uint8)
    for c in result.rejected:
        flat[c.pixels] = (0.7 * flat[c.pixels]).astype(np.uint8)
    img = flat.reshape(h, w, 3)
    for k, c in enumerate(result.objects):
        r, u = np.unravel_index(c.pixels, (h, w))
        label = f"{c.track_id}{'*' if c.partially_visible else ''}"
        cv2.putText(img, label, (int(u.mean()) - 15, int(r.mean())), 0, 0.42, (0, 0, 0), 3)
        cv2.putText(img, label, (int(u.mean()) - 15, int(r.mean())), 0, 0.42, (255, 255, 255), 1)
    for c in result.rejected:
        r, u = np.unravel_index(c.pixels, (h, w))
        cv2.putText(img, c.rejected, (int(u.mean()) - 15, int(r.mean())), 0, 0.4, (0, 0, 255), 1)
    cv2.putText(img, title, (5, 18), 0, 0.5, (0, 255, 255), 1)
    return img


def fmt(c):
    return (f"{c.track_id or c.rejected:>9} xyz ({c.center[0]:+.3f} {c.center[1]:+.3f} {c.center[2]:+.3f}) yaw {c.yaw:+.2f} "
            f"size {c.size[0]:.3f} {c.size[1]:.3f} {c.size[2]:.3f} top {c.top_z:+.3f} low {c.lowest:.3f} "
            f"floor {c.floor_frac:.2f} n {c.n_points}{' CUT' if c.partially_visible else ''}")


def table_area_edges(cloud, plane):
    """x range of the visible table top in pelvis, by rows of plane inliers: for checking NEAR_EDGE_X."""
    pts = cloud.reshape(-1, 3)
    with np.errstate(invalid="ignore"):
        on = np.abs(plane.height(pts)) < 0.002
    return np.percentile(pts[on, 0], [0.05, 99.95])


def floor_plane(cloud):
    """Least squares over points of the floor, as an independent check of the tilt. None if too few are visible."""
    pts = cloud.reshape(-1, 3)
    with np.errstate(invalid="ignore"):
        q = pts[(pts[:, 2] < -0.5) & (pts[:, 2] > -1.0) & (pts[:, 0] > 1.2) & (np.abs(pts[:, 1]) < 1.0)]
    if len(q) < 2000:
        return None
    a = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
    for _ in range(3):
        coef = np.linalg.lstsq(a, q[:, 2], rcond=None)[0]
        keep = np.abs(a @ coef - q[:, 2]) < 0.01
        a, q = a[keep], q[keep]
    return coef


def report_consistency(views, cam):
    """Same track seen from several poses: spread of the centre and top_z, whole views apart from cut ones."""
    print(f"consistency, camera {cam} (mm, max over pairs of poses):")
    worst = {False: [0.0, 0.0], True: [0.0, 0.0]}
    for layout, tracks in sorted(views.items()):
        for track, seen in sorted(tracks.items()):
            if len(seen) < 2:
                continue
            line, poses = [], list(seen)
            for i, a in enumerate(poses):
                for b in poses[i + 1:]:
                    ca, cb = seen[a], seen[b]
                    d = float(np.hypot(*(ca.center[:2] - cb.center[:2]))) * 1e3
                    dz = abs(ca.top_z - cb.top_z) * 1e3
                    cut = ca.partially_visible or cb.partially_visible
                    worst[cut][0] = max(worst[cut][0], d)
                    worst[cut][1] = max(worst[cut][1], dz)
                    line.append(f"{a[0]}-{b[0]}{'(cut)' if cut else ''} {d:.1f}/{dz:.1f}")
            print(f"  {layout} {track}: " + "  ".join(line))
    print(f"  worst whole views: centre {worst[False][0]:.1f} top_z {worst[False][1]:.1f}; "
          f"with a cut view: centre {worst[True][0]:.1f} top_z {worst[True][1]:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--poses", default="center,left,right")
    ap.add_argument("--cams", default="head,torso")
    ap.add_argument("--quiet", action="store_true", help="only the per-frame lines")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    times = {}
    views = {}  # camera -> layout -> track -> pose -> cluster
    for layout in sorted(glob.glob(os.path.join(args.frames, "layout*"))):
        name = os.path.basename(layout)
        tracker = tc.Tracker()  # one per layout: the objects stay put between the poses
        for pose in args.poses.split(","):
            for cam in args.cams.split(","):
                path = os.path.join(layout, f"{pose}_cam_{cam}.npz")
                if not os.path.exists(path):
                    continue
                d = np.load(path)
                t0 = time.perf_counter()
                cloud = cloud_from_frame(d)
                t1 = time.perf_counter()
                result = tc.detect(cloud)
                tracker.update(result.objects) if result else None
                t2 = time.perf_counter()
                times.setdefault(cam, []).append((t1 - t0, t2 - t1))
                if result is None:
                    print(f"{name} {pose} {cam}: no table")
                    continue
                p = result.plane
                edges = table_area_edges(cloud, p)
                floor = floor_plane(cloud)
                floor_txt = ""
                if floor is not None:  # scene.yaml: the top is 0.70 above the floor
                    rise = 0.70 * np.sqrt(1 + floor[0] ** 2 + floor[1] ** 2)
                    from_floor = floor[0] * tc.TABLE_REF_XY[0] + floor[1] * tc.TABLE_REF_XY[1] + floor[2] + rise
                    floor_txt = f" floor z0 {floor[2]:.3f} slope {floor[0]:.4f} table_z_from_floor {from_floor:.4f}"
                for c in result.objects:
                    views.setdefault(cam, {}).setdefault(name, {}).setdefault(c.track_id, {})[pose] = c
                print(f"{name} {pose} {cam}: table_z {result.table_z:.4f} z(0,0) {p.z_at(0, 0):.4f} "
                      f"slope {-p.normal[0] / p.normal[2]:.4f} rms {p.rms * 1e3:.2f}mm inliers {p.n_inliers} "
                      f"x {edges[0]:.3f}..{edges[1]:.3f}{floor_txt} objects {len(result.objects)} "
                      f"rejected {[c.rejected for c in result.rejected]} {(t2 - t0) * 1e3:.0f} ms")
                if not args.quiet:
                    for c in result.objects + result.rejected:
                        print("   ", fmt(c))
                cv2.imwrite(os.path.join(args.out, f"{name}_{pose}_{cam}.jpg"),
                            draw(d["rgb"], result, f"{name} {pose} {cam}: {len(result.objects)} objects"))
    for cam, by_layout in views.items():
        report_consistency(by_layout, cam)
    for cam, ts in times.items():
        a = np.array(ts) * 1e3
        total = a.sum(axis=1)
        print(f"time {cam}: cloud {a[:, 0].mean():.0f} ms, detect {a[:, 1].mean():.0f} ms, "
              f"total mean {total.mean():.0f} max {total.max():.0f} ms over {len(a)} frames")


if __name__ == "__main__":
    main()
