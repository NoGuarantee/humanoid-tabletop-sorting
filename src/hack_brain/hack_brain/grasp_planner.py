"""Grasp and place planning from detected objects. Owner: grasping. Pure Python, no ROS node.

Coordinates: `pelvis` frame. Parameters per object/category come from config/grasp_db.yaml.
"""
import math
from dataclasses import dataclass

import yaml
from geometry_msgs.msg import Point, Pose, Quaternion

# URDF/TF mount-to-fingertip distance; also used by perception arm masking.
FINGERTIP = 0.21


@dataclass
class GraspCandidate:
    arm: str  # left | right
    pre_grasp: Pose  # above the object
    grasp: Pose  # at the grasp point
    lift: Pose  # after lifting
    hand_preset: str
    force: float
    score: float


@dataclass
class PlacePlan:
    arm: str
    approach: Pose  # above the tray / slot
    release: Pose  # where the hand opens
    retreat: Pose
    waist_yaw: float  # 0 at the table, 1.57 at the fume hood


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return Quaternion(x=sr * cp * cy - cr * sp * sy, y=cr * sp * cy + sr * cp * sy,
                      z=cr * cp * sy - sr * sp * cy, w=cr * cp * cy + sr * sp * sy)


def top_down(yaw):
    """TCP orientation for a grasp from above: fingers down, closing across `yaw`.

    Checked against the URDF and tf: the TCP sits at the hand mount, the fingers run along its -z (tips 0.21 m away)
    and the palm faces +y on the right hand, -y on the left. So the TCP z axis points up, as on the hanging arm, and
    the thumb closes on the fingers along y. Rotation by pi about x would point the fingers at the ceiling.
    """
    return quat_from_rpy(0.0, 0.0, yaw)


def pose(x, y, z, q):
    return Pose(position=Point(x=float(x), y=float(y), z=float(z)), orientation=q)


def tcp_at(x, y, z, q):
    """Convert a fingertip goal to the hand-mount TCP (same convention as pick-one)."""
    ax = 2 * (q.x*q.z + q.w*q.y)
    ay = 2 * (q.y*q.z - q.w*q.x)
    az = 1 - 2 * (q.x*q.x + q.y*q.y)
    return pose(x+FINGERTIP*ax, y+FINGERTIP*ay, z+FINGERTIP*az, q)


def load_db(path):
    with open(path) as f:
        return yaml.safe_load(f)


def params_for(db, name, category):
    p = dict(db.get("default", {}))
    p.update(db.get("by_category", {}).get(category, {}))
    p.update(db.get("by_name", {}).get(name, {}))
    return p


def choose_arm(obj):
    """Objects on the left (y > 0) with the left arm."""
    return "left" if obj.center.y > 0 else "right"


def plan_grasps(obj, scene, db):
    """Candidates for one DetectedObject, best first."""
    p = params_for(db, obj.name, obj.category)
    arm = choose_arm(obj)
    x, y = obj.center.x, obj.center.y
    local_table = obj.top_z - obj.size.z if obj.size.z > 0 else scene.table_z
    z = max(obj.top_z + p.get("z_offset", -0.03),
            max(scene.table_z, local_table) + p.get("min_above_table", 0.015))
    base_yaw = obj.yaw + (0.0 if p.get("align", "short_axis") == "short_axis" else math.pi / 2)
    out = []
    for i, dyaw in enumerate((0.0, math.pi / 2)):  # second candidate: rotated hand, used on a retry
        q = top_down(base_yaw + dyaw)
        out.append(GraspCandidate(
            arm=arm,
            pre_grasp=tcp_at(x, y, z + p.get("pre_height", 0.10), q),
            grasp=tcp_at(x, y, z, q),
            lift=tcp_at(x, y, z + p.get("lift_height", 0.15), q),
            hand_preset=p.get("preset", "power"),
            force=p.get("force", 0.0),
            score=1.0 - 0.5 * i,
        ))
    return out


def plan_place(arm, container, db):
    """Drop above the centre of a tray. Tubes into slots: TODO (milestone M4)."""
    p = db.get("place", {})
    c = container.pose.position
    q = top_down(0.0)
    above = c.z + p.get("approach_height", 0.20)
    return PlacePlan(arm=arm,
                     approach=tcp_at(c.x, c.y, above, q),
                     release=tcp_at(c.x, c.y, c.z + p.get("release_height", 0.10), q),
                     retreat=tcp_at(c.x, c.y, above, q),
                     waist_yaw=0.0)
