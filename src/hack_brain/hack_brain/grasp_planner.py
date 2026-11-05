"""Grasp and place planning from detected objects. Owner: grasping. Pure Python, no ROS node.

Coordinates: `pelvis` frame. Parameters per object/category come from config/grasp_db.yaml.
Heights in the database are fingertip heights. The TCP is the palm: the tips lie `FINGERTIP` metres
along its -z, so every pose sent to motion is shifted along TCP +z (see `tcp_at`).
"""
import math
from dataclasses import dataclass

import yaml
from geometry_msgs.msg import Point, Pose, Quaternion

# Along TCP -z from the mount to the tips. Same length as perception_node.FINGERS; top_down points this axis down.
FINGERTIP = 0.21


@dataclass
class GraspCandidate:
    arm: str  # left | right
    pre_grasp: Pose  # TCP, fingertips one pre_height above the grasp
    grasp: Pose  # TCP, fingertips at the grasp point
    lift: Pose  # TCP, fingertips one lift_height above the grasp
    hand_preset: str
    force: float
    score: float


@dataclass
class PlacePlan:
    arm: str
    approach: Pose  # TCP, fingertips approach_height above the opening
    release: Pose  # TCP, where the hand opens
    retreat: Pose
    waist_yaw: float  # 0 at the table, 1.57 at the fume hood


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return Quaternion(x=sr * cp * cy - cr * sp * sy, y=cr * sp * cy + sr * cp * sy,
                      z=cr * cp * sy - sr * sp * cy, w=cr * cp * cy + sr * sp * sy)


def top_down(yaw):
    """TCP orientation for a grasp from above: fingers down, TCP +x along `yaw`.

    Checked against the URDF and tf: the TCP sits at the hand mount, the fingers run along its -z (tips 0.21 m away)
    and the palm faces +y on the right hand, -y on the left. So the TCP z axis points up, as on the hanging arm, and
    the thumb closes on the fingers along +y, which is `yaw + π/2` in the table plane. Rotation by pi about x
    would point the fingers at the ceiling.
    """
    return quat_from_rpy(0.0, 0.0, yaw)


def pinch_yaw(obj_yaw, align):
    """`top_down` argument so the hand closes across the short side, or the long side when asked.

    The object's yaw is its long axis. Closing is TCP +y, a right angle ahead of the `top_down` yaw, so the
    yaw that pinches the short side is the object's own yaw. Adding π/2 here as well points the pinch along
    the long side: on seed 1 the first candidate then met a 7.7 cm face and stalled.
    """
    if align != "short_axis":
        return obj_yaw + math.pi / 2
    return obj_yaw


def pose(x, y, z, q):
    return Pose(position=Point(x=float(x), y=float(y), z=float(z)), orientation=q)


def axis_z(q):
    """Where the TCP +z axis points, for a geometry_msgs Quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def tcp_at(x, y, z, q):
    """TCP pose whose fingertips are at (x, y, z). The tips lie FINGERTIP metres along TCP -z."""
    ax, ay, az = axis_z(q)
    return pose(x + FINGERTIP * ax, y + FINGERTIP * ay, z + FINGERTIP * az, q)


def fingertip_z(obj, scene, p):
    """Height of the fingertips for a top-down grasp.

    `scene.table_z` is the plane at one reference point and the top rises toward +x, so the floor is the
    table under this object (`top_z - size.z`, see table_clusters), and never below that reference.
    """
    local = obj.top_z - obj.size.z if obj.size.z > 0 else scene.table_z
    floor = max(scene.table_z, local) + p.get("min_above_table", 0.015)
    return max(obj.top_z + p.get("z_offset", -0.03), floor)


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


def grasp_xy(obj, tcp_yaw, preset):
    """Where the fingertips close.

    The palm centre is what `tcp_at` aims at. A pinch there closes over the middle, and a flat dish is only
    a rim at that height, so the fingers meet nothing. Aim just outside the near rim: the thumb closes
    along TCP +y and meets the wall.
    """
    x, y = obj.center.x, obj.center.y
    if preset != "pinch":
        return x, y
    delta = (tcp_yaw + math.pi / 2.0) - obj.yaw
    across = abs(math.cos(delta)) * obj.size.x + abs(math.sin(delta)) * obj.size.y
    # Flat: 1 cm and 2 cm outside both left the open index on the top (the reading stayed near 1.1, then
    # the finger closed fully once the hand lifted). A wrist turned on its side at that height has no IK.
    # Taller than 5 cm: the tips are already down the side, so the point stays on the rim itself.
    outside = 0.02 if obj.size.z < 0.05 else 0.0
    shift = 0.5 * across + outside
    dx, dy = -math.sin(tcp_yaw), math.cos(tcp_yaw)  # TCP +y
    near = (x - shift * dx, y - shift * dy)
    far = (x + shift * dx, y + shift * dy)
    return near if math.hypot(*near) <= math.hypot(*far) else far


def plan_grasps(obj, scene, db):
    """Candidates for one DetectedObject, best first."""
    p = params_for(db, obj.name, obj.category)
    arm = choose_arm(obj)
    if p.get("preset", "power") == "pinch" and obj.size.z > 0.05:
        p = dict(p)  # a dish/figure taller than the fingers' skim: close on the side, not the top face
        p["z_offset"] = -min(0.04, 0.4 * obj.size.z)
    z = fingertip_z(obj, scene, p)  # tips, not the TCP: tcp_at lifts the palm by FINGERTIP
    base_yaw = pinch_yaw(obj.yaw, p.get("align", "short_axis"))
    out = []
    for i, dyaw in enumerate((0.0, math.pi / 2)):  # second candidate: rotated hand, used on a retry
        yaw = base_yaw + dyaw
        q = top_down(yaw)
        x, y = grasp_xy(obj, yaw, p.get("preset", "power"))
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
    above = c.z + p.get("approach_height", 0.20)  # fingertips above the opening; tcp_at raises the palm
    return PlacePlan(arm=arm,
                     approach=tcp_at(c.x, c.y, above, q),
                     release=tcp_at(c.x, c.y, c.z + p.get("release_height", 0.10), q),
                     retreat=tcp_at(c.x, c.y, above, q),
                     waist_yaw=0.0)
