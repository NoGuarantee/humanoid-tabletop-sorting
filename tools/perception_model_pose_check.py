#!/usr/bin/env python3
"""Call the experimental service and check its schema and source-frame freshness."""
import argparse
import json
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

SERVICE = "/perception/estimate_model_poses"
SCHEMA = "hack_perception.model_poses/v1"


def check(node, max_age_s, timeout_s, require_pose):
    client = node.create_client(Trigger, SERVICE)
    if not client.wait_for_service(timeout_sec=timeout_s):
        raise RuntimeError(f"{SERVICE} is unavailable")
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        raise RuntimeError(f"{SERVICE} timed out")
    result = future.result()
    data = json.loads(result.message)
    if data["schema"] != SCHEMA or data["frame_id"] != "pelvis":
        raise RuntimeError("unexpected model pose response schema or frame")
    if not result.success or data["status"] != "ok":
        raise RuntimeError(f"service declined the request: {data['status']}")
    age_s = (node.get_clock().now().nanoseconds - data["stamp_ns"]) / 1e9
    if not 0 <= age_s <= max_age_s:
        raise RuntimeError(f"source frame age {age_s:.3f}s exceeds caller's limit {max_age_s:.3f}s")
    accepted = []
    for obj in data["objects"]:
        matrix = obj["T_pelvis_object"]
        if obj["pose_status"] == "ok":
            if matrix is None or len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                raise RuntimeError("accepted object has no 4x4 model transform")
            accepted.append(obj["name"])
        elif matrix is not None:
            raise RuntimeError("refused object leaked a model transform")
    if require_pose and not accepted:
        raise RuntimeError("no accepted model pose on this exposure")
    print(json.dumps(dict(age_s=age_s, accepted=accepted, response=data), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-s", type=float, required=True, help="maximum age of the exposure in sim seconds")
    parser.add_argument("--timeout-s", type=float, default=15., help="wall-time service wait/call limit")
    parser.add_argument("--require-pose", action="store_true", help="require at least one accepted object")
    args = parser.parse_args()
    if not args.max_age_s > 0 or not args.timeout_s > 0:
        parser.error("age and timeout must be positive")
    rclpy.init(args=[])
    node = Node("perception_model_pose_check", parameter_overrides=[
        Parameter("use_sim_time", Parameter.Type.BOOL, True)
    ])
    try:
        check(node, args.max_age_s, args.timeout_s, args.require_pose)
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
