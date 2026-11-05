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
    approach_preset: str = "open"  # commanded at pre_grasp, before the descent


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


def rpy_matrix(roll, pitch, yaw):
    """Rotation of the TCP frame, same convention as `quat_from_rpy`: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def matvec(m, v):
    return tuple(m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3))


def matrix_from_quat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


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


def span_across(obj, tcp_yaw):
    """Object width along the pinch, which is TCP +y (a right angle ahead of `tcp_yaw`)."""
    delta = (tcp_yaw + math.pi / 2.0) - obj.yaw
    return abs(math.cos(delta)) * obj.size.x + abs(math.sin(delta)) * obj.size.y


def grasp_xy(obj, tcp_yaw, preset):
    """Where the fingertips close.

    The palm centre is what `tcp_at` aims at. A pinch there closes over the middle, and a dish is only a rim
    at that height, so the fingers meet nothing. Aim at the near rim: the thumb closes along TCP +y and meets
    the wall. Flat objects (under 5 cm) do not use this; see `plan_flat_pinches`.
    """
    x, y = obj.center.x, obj.center.y
    if preset != "pinch":
        return x, y
    # The open index tip lies 2.2 cm toward TCP +y from the point tcp_at aims at. Aiming the axis at the
    # rim puts that tip inside the object, and the close pushes the finger past open (a reading above 1).
    shift = 0.5 * span_across(obj, tcp_yaw) + 0.03
    dx, dy = -math.sin(tcp_yaw), math.cos(tcp_yaw)  # TCP +y
    near = (x - shift * dx, y - shift * dy)
    far = (x + shift * dx, y + shift * dy)
    return near if math.hypot(*near) <= math.hypot(*far) else far


# Ready fingertips in the right-hand TCP frame: fingers open, thumb rotated to actuator 6 = 0.1, thumb bend open.
# The left hand mirrors y. At 45° the two pads are level and 9.8 cm apart; the index closes onto the thumb.
# 55° left the thumb 1.7 cm lower, and on a 3 cm dish that pad was on the table (bend reading 1.34, past open).
# A fully sideways hand (wrist at the table, z ≈ -0.14) has no IK.
FLAT_INDEX = (0.0294, 0.0220, -0.2088)
FLAT_THUMB = (0.0365, 0.0913, -0.1401)
FLAT_ROLL = math.radians(45)
FLAT_APERTURE = 0.098  # tip-to-tip distance of the ready hand
FLAT_CLEAR = 0.015  # spare across the gap, so a small miss still leaves both tips outside


def flat_locals(arm):
    """(index, thumb) in the TCP frame. The thumb is on +y for the right hand and -y for the left."""
    sign = 1.0 if arm == "right" else -1.0
    index = (FLAT_INDEX[0], sign * FLAT_INDEX[1], FLAT_INDEX[2])
    thumb = (FLAT_THUMB[0], sign * FLAT_THUMB[1], FLAT_THUMB[2])
    return index, thumb


def flat_hold_point(arm):
    """TCP-frame point that stays on the object while a flat pinch carries it."""
    index, thumb = flat_locals(arm)
    return tuple((a + b) / 2.0 for a, b in zip(index, thumb))


def tcp_from_local(target, local, q):
    """TCP pose that puts the TCP-frame point `local` at the world point `target`."""
    off = matvec(matrix_from_quat(q), local)
    return pose(target[0] - off[0], target[1] - off[1], target[2] - off[2], q)


def _flat_tip_z(obj, scene, span):
    """World z of the lower ready tip. `span` is how much higher the other tip is.

    The lower tip stays off the table. The upper one is kept just under the top, so it meets the wall
    and not the face above it. A shorter object cannot fit both; the lower tip still clears the table.
    """
    local = obj.top_z - obj.size.z if obj.size.z > 0 else scene.table_z
    floor = max(scene.table_z, local)
    lower = floor + 0.018  # the index dips 8 mm while it curls; this keeps the pad off the table
    # A tall object is held at mid-height. The bottom leaves the mass above the pads and it topples out.
    mid = (floor + obj.top_z) / 2.0 - span / 2.0
    if mid > lower:
        lower = mid
    if lower + span > obj.top_z - 0.004:
        lower = obj.top_z - 0.004 - span
    if lower < floor + 0.016:
        lower = floor + 0.016
    return lower


def plan_flat_pinches(obj, scene, p, arm):
    """Pinch a flat object with the palm leaned over, thumb already across from the index.

    Upright, the thumb pad is 5 cm above the open index. On a 3 cm dish the index curls into the top
    (it stays near 1.0, then finishes to 0.3 once the hand lifts) and the thumb closes on nothing.
    Leaning 45° puts the two pads level, on opposite faces. The wrist stays around z = 0, which IK can reach;
    the sideways pose with the wrist on the table cannot.
    """
    pre_h = p.get("pre_height", 0.10)
    lift_h = p.get("lift_height", 0.15)
    base_yaw = pinch_yaw(obj.yaw, p.get("align", "short_axis"))
    out = []
    for i, dyaw in enumerate((0.0, math.pi / 2)):
        yaw = base_yaw + dyaw
        if span_across(obj, yaw) + FLAT_CLEAR > FLAT_APERTURE:
            continue  # wider than the open fingers: a tip would start inside the object
        sign = 1.0 if arm == "right" else -1.0
        roll = -sign * FLAT_ROLL
        index, thumb = flat_locals(arm)
        rot = rpy_matrix(roll, 0.0, yaw)
        i_w, t_w = matvec(rot, index), matvec(rot, thumb)
        span = abs(i_w[2] - t_w[2])
        lower = _flat_tip_z(obj, scene, span)
        thumb_z = lower if t_w[2] <= i_w[2] else lower + span
        mid = tuple((a + b) / 2.0 for a, b in zip(i_w, t_w))
        tcp = (obj.center.x - mid[0], obj.center.y - mid[1], thumb_z - t_w[2])
        q = quat_from_rpy(roll, 0.0, yaw)
        # Straight down, not along the fingers. At 45° that axis runs through the dish: the thumb
        # sweeps it on the way in and the bend is pushed past open (the reading goes to 1.33).
        pre = (tcp[0], tcp[1], tcp[2] + pre_h)
        out.append(GraspCandidate(
            arm=arm,
            pre_grasp=pose(*pre, q),
            grasp=pose(*tcp, q),
            lift=pose(tcp[0], tcp[1], tcp[2] + lift_h, q),  # straight up: along the fingers would drag it
            hand_preset="pinch_flat",
            force=p.get("force", 0.0),
            score=1.0 - 0.5 * i,
            approach_preset="pinch_ready",
        ))
    return out


def plan_grasps(obj, scene, db):
    """Candidates for one DetectedObject, best first."""
    p = params_for(db, obj.name, obj.category)
    arm = choose_arm(obj)
    if p.get("preset", "power") == "pinch" and obj.size.z <= 0.05:
        # The upright pinch meets 5 cm above the open index, so the thumb closes above a short top and the
        # index stays open against the face. The leaned hand spans 9.8 cm with both pads level. A taller
        # object uses the upright rim grasp: the leaned pre-grasp over a 13 cm box had no IK (197 mm).
        flat = plan_flat_pinches(obj, scene, p, arm)
        if flat:
            return flat
    if p.get("preset", "power") == "pinch" and obj.size.z > 0.05:
        p = dict(p)  # the thumb pad is 5 cm above the open index; keep that pad below the top
        p["z_offset"] = -min(0.5 * obj.size.z, max(0.06, 0.4 * obj.size.z))
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
