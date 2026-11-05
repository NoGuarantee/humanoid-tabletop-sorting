#!/usr/bin/env python3
"""Smoke-test the normal Perceive service and optional model poses on a ROS stand."""
import argparse
import json
import sys
from time import monotonic, sleep

from hack_interfaces.srv import Perceive
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

POSE_SERVICE = "/perception/estimate_model_poses"
SCENE_SERVICE = "/perception/perceive"
SCHEMA = "hack_perception.model_poses/v1"


def call(node, client, request, deadline):
    start = monotonic()
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=max(0., deadline - start))
    if not future.done():
        raise RuntimeError(f"{client.srv_name} timed out")
    return future.result(), monotonic() - start


def scene_sample(node, scene_client, deadline, args):
    result, elapsed = call(node, scene_client, Perceive.Request(camera="head", region="table"), deadline)
    if not result.success or result.scene.header.frame_id != "pelvis":
        raise RuntimeError(f"Perceive failed: {result.message}")
    stamp = result.scene.header.stamp.sec * 10**9 + result.scene.header.stamp.nanosec
    age_s = (node.get_clock().now().nanoseconds - stamp) / 1e9
    if stamp <= 0 or not 0 <= age_s <= 1.:
        raise RuntimeError(f"Perceive has an invalid or stale source stamp: {stamp}, age {age_s:.3f}s")
    if elapsed > args.scene_max_wall_s:
        raise RuntimeError(f"Perceive took {elapsed:.3f}s, budget is {args.scene_max_wall_s:.3f}s")
    trays = [c.id for c in result.scene.containers if c.kind == "tray"]
    return dict(stamp_ns=stamp, age_s=age_s, wall_s=elapsed, table_z_m=result.scene.table_z, trays=trays)


def check(node, args):
    scene_client = node.create_client(Perceive, SCENE_SERVICE)
    pose_client = None if args.scene_only else node.create_client(Trigger, POSE_SERVICE)
    if not scene_client.wait_for_service(timeout_sec=args.timeout_s):
        raise RuntimeError(f"{SCENE_SERVICE} is unavailable")
    deadline = monotonic() + args.timeout_s
    first = scene_sample(node, scene_client, deadline, args)
    print(json.dumps(dict(first_scene=first)), flush=True)
    if args.require_first_tray and not first["trays"]:
        raise RuntimeError("first Perceive response contains no trays")
    scenes = [first]

    data, age_s = None, None
    if pose_client is not None:
        if not pose_client.wait_for_service(timeout_sec=max(0., deadline - monotonic())):
            raise RuntimeError(f"{POSE_SERVICE} is unavailable")
        last_status = None
        while monotonic() < deadline:
            result, _ = call(node, pose_client, Trigger.Request(), deadline)
            data = json.loads(result.message)
            last_status = data["status"]
            if data["schema"] != SCHEMA or data["frame_id"] != "pelvis":
                raise RuntimeError("unexpected model pose response schema or frame")
            if result.success and last_status == "ok":
                age_s = (node.get_clock().now().nanoseconds - data["stamp_ns"]) / 1e9
                if 0 <= age_s <= args.max_age_s and data["stamp_ns"] > args.after_stamp_ns:
                    break
            elif last_status not in {"worker_loading", "awaiting_fresh_pose"}:
                raise RuntimeError(f"model pose request declined: {last_status}")
            scenes.append(scene_sample(node, scene_client, deadline, args))
            sleep(.08)
        else:
            raise RuntimeError(f"no fresh model pose response within timeout ({last_status})")

    while len(scenes) < args.scene_samples:
        scenes.append(scene_sample(node, scene_client, deadline, args))
        sleep(.05)
    accepted = []
    for obj in data["objects"] if data is not None else ():
        matrix = obj["T_pelvis_object"]
        if obj["pose_status"] == "ok":
            if matrix is None or len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                raise RuntimeError("accepted object has no 4x4 model transform")
            accepted.append(obj["name"])
        elif matrix is not None:
            raise RuntimeError("refused object leaked a model transform")
    if args.require_pose and not accepted:
        raise RuntimeError("no accepted model pose on this exposure")
    print(json.dumps(dict(pose_age_s=age_s, accepted=accepted, scenes=scenes, response=data), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-s", type=float, default=1.75, help="caller budget for a pose in sim seconds")
    parser.add_argument("--after-stamp-ns", type=int, default=0, help="require a newer pose exposure")
    parser.add_argument("--timeout-s", type=float, default=15., help="wall-time service wait/call limit")
    parser.add_argument("--scene-only", action="store_true", help="probe only the standard Perceive service")
    parser.add_argument("--scene-samples", type=int, default=8)
    parser.add_argument("--scene-max-wall-s", type=float, default=.5)
    parser.add_argument("--require-first-tray", action="store_true", help="fail if first Perceive returns no tray")
    parser.add_argument("--require-pose", action="store_true", help="require at least one accepted object")
    args = parser.parse_args()
    if not all(value > 0 for value in (args.max_age_s, args.timeout_s, args.scene_max_wall_s, args.scene_samples)):
        parser.error("age, timeout, latency and sample count must be positive")
    if args.scene_only and args.require_pose:
        parser.error("--require-pose requires model poses")
    rclpy.init(args=[])
    node = Node("perception_model_pose_check", parameter_overrides=[
        Parameter("use_sim_time", Parameter.Type.BOOL, True)
    ])
    try:
        check(node, args)
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
