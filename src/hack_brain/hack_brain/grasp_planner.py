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
    the wall. Objects no taller than FLAT_MAX_HEIGHT use `plan_flat_pinches` instead.
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
# right_index_1 origin in that same frame. It sits behind the pads; the pinch is placed on the far side of it.
FLAT_KNUCKLE = (0.020493, -0.000285, -0.136914)
FLAT_ROLL = math.radians(45)
FLAT_APERTURE = 0.098  # tip-to-tip distance of the ready hand
# The soap dish's short side equals this aperture, so both pads cannot sit outside the box.
# The chord that the parked arm can reach leaves the thumb 3 mm inside the box (bend reading
# 1.22, past open) and the index just outside, so the index closes to the preset without
# meeting the wall. Putting the index 12 mm inside the box and the thumb 12 mm outside — on
# the wall — has no collision-free path. INDEX_NEAR is the index clearance (negative = inside
# the box), THUMB_OUT the least clearance accepted for the thumb.
INDEX_NEAR = (-0.016, 0.006)
THUMB_OUT = -0.004
# pinch_flat closes to about 3 cm. Less dish than that between the open pads, and the index
# reaches the preset without ever touching the wall.
MIN_BETWEEN = 0.040
# Below this the upright thumb pad, 7 cm above the open index, misses the wall. The soap dish is 5.1 cm.
FLAT_MAX_HEIGHT = 0.065


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


def _xy_unit(v):
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return (1.0, 0.0)
    return (v[0] / n, v[1] / n)


def _box_clear(obj, xy):
    """Metres outside the object's box in the table plane. Negative when `xy` is inside.

    `xy` is relative to the centre. The box is the mesh, not the ellipse inscribed in it: on the soap
    dish the ellipse left the thumb 3 mm inside the real wall.
    """
    c, s = math.cos(obj.yaw), math.sin(obj.yaw)
    x = c * xy[0] + s * xy[1]
    y = -s * xy[0] + c * xy[1]
    dx = abs(x) - max(obj.size.x, 1e-3) / 2.0
    dy = abs(y) - max(obj.size.y, 1e-3) / 2.0
    if dx <= 0.0 and dy <= 0.0:
        return -min(-dx, -dy)
    if dx > 0.0 and dy > 0.0:
        return math.hypot(dx, dy)
    return max(dx, dy)


def _between(obj, a, b):
    """Length of the segment that lies inside the box."""
    n = 40
    hit = 0
    for i in range(n + 1):
        t = i / n
        if _box_clear(obj, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))) <= 0.0:
            hit += 1
    return hit / n * math.hypot(b[0] - a[0], b[1] - a[1])


def _flat_pad_z(obj, scene):
    """Height of the level pads: under the top, and above the 8 mm dip of the closing index."""
    local = obj.top_z - obj.size.z if obj.size.z > 0 else scene.table_z
    floor = max(scene.table_z, local)
    low = floor + 0.020
    high = obj.top_z - 0.010
    if high < floor + 0.012:
        return None
    return min(max(0.5 * (floor + obj.top_z), low), high)


def _finger_on_top(obj, index_xy, pad_z, i_w, k_w, hand):
    """The open index crosses the top face, so a straight descent would set the finger down on it.

    `index_xy` is the pad relative to the centre, `hand` points out of the dish. The finger is about
    1 cm thick on the inner side of that line.
    """
    for a in (0.0, 0.2, 0.4, 0.6, 0.8):
        if pad_z + a * (k_w[2] - i_w[2]) > obj.top_z + 0.004:
            continue
        p = (index_xy[0] + a * (k_w[0] - i_w[0]), index_xy[1] + a * (k_w[1] - i_w[1]))
        belly = (p[0] - 0.008 * hand[0], p[1] - 0.008 * hand[1])
        # The mesh box overstates a hollow dish. Under 2 cm into the box is still the rim; deeper,
        # the descent sets the finger down on the face.
        if _box_clear(obj, p) < -0.020 or _box_clear(obj, belly) < -0.020:
            return True
    return False


def plan_flat_pinches(obj, scene, p, arm):
    """Pinch a flat object with the palm leaned 45°, across a corner the open fingers can span.

    Upright, the thumb pad is 7 cm above the open index, so on a short dish the index curls onto the
    top and only finishes closing after the lift. Leaning 45° puts the pads level. The aperture is
    9.8 cm and the soap dish's short side is the same, so a chord straight across that side starts
    with one pad inside the wall. The pinch crosses a corner: the index a few millimetres outside
    its wall, the thumb clear of the other, and enough of the dish between them that the closing
    index meets the wall before the preset. The wrist stays near z = 0, which IK can reach; the
    sideways pose with the wrist on the table cannot.
    """
    pad_z = _flat_pad_z(obj, scene)
    if pad_z is None:
        return []
    pre_h = p.get("pre_height", 0.10)
    lift_h = p.get("lift_height", 0.15)
    base_yaw = pinch_yaw(obj.yaw, p.get("align", "short_axis"))
    sign = 1.0 if arm == "right" else -1.0
    roll = -sign * FLAT_ROLL
    index, thumb = flat_locals(arm)
    knuckle = (FLAT_KNUCKLE[0], sign * FLAT_KNUCKLE[1], FLAT_KNUCKLE[2])
    found = []
    for step in range(-8, 9):
        yaw = base_yaw + step * math.radians(10)
        rot = rpy_matrix(roll, 0.0, yaw)
        i_w, t_w, k_w = matvec(rot, index), matvec(rot, thumb), matvec(rot, knuckle)
        gap_v = (t_w[0] - i_w[0], t_w[1] - i_w[1])
        gap = math.hypot(*gap_v)
        if gap < 0.05:
            continue
        pinch = (gap_v[0] / gap, gap_v[1] / gap)
        side = (k_w[0] - i_w[0], k_w[1] - i_w[1])
        along = side[0] * pinch[0] + side[1] * pinch[1]
        hand = _xy_unit((side[0] - along * pinch[0], side[1] - along * pinch[1]))
        best = None
        for hs in range(-32, 33):
            for ps in range(-56, 57):
                h, s = hs * 0.005, ps * 0.005
                idx = (h * hand[0] + s * pinch[0], h * hand[1] + s * pinch[1])
                thu = (idx[0] + gap * pinch[0], idx[1] + gap * pinch[1])
                near, far = _box_clear(obj, idx), _box_clear(obj, thu)
                if not (INDEX_NEAR[0] <= near <= INDEX_NEAR[1] and far >= THUMB_OUT):
                    continue
                span = _between(obj, idx, thu)
                tcp_x = obj.center.x + idx[0] - i_w[0]
                tcp_y = obj.center.y + idx[1] - i_w[1]
                # x 0.26 stalled 15 cm short. The pinch at (0.32, -0.10) is the one that reached.
                if span < MIN_BETWEEN or tcp_x < 0.30:
                    continue
                if _finger_on_top(obj, idx, pad_z, i_w, k_w, hand):
                    continue
                reach = -math.hypot(tcp_x - 0.317, tcp_y + 0.100)
                rank = span + 0.4 * reach
                if best is None or rank > best[0]:
                    best = (rank, span, reach, idx, i_w, yaw)
        if best is not None:
            found.append(best)
    def _rank(item):
        _inner, span, reach, _idx, _iw, yaw = item
        # The parked arm reaches a pinch across the short side. A freer corner does not.
        dy = abs((yaw - base_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        return -(span - 0.25 * dy + 0.4 * reach)

    found.sort(key=_rank)
    out = []
    yaws = []
    for _inner, span, _reach, index_xy, i_w, yaw in found:
        if any(abs((yaw - prev + math.pi) % (2 * math.pi) - math.pi) < 0.4 for prev in yaws):
            continue  # the retry is a different corner
        q = quat_from_rpy(roll, 0.0, yaw)
        tcp = (obj.center.x + index_xy[0] - i_w[0], obj.center.y + index_xy[1] - i_w[1], pad_z - i_w[2])
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
            score=span,
            approach_preset="pinch_ready",
        ))
        yaws.append(yaw)
        if len(out) == 2:
            break
    return out


def plan_grasps(obj, scene, db):
    """Candidates for one DetectedObject, best first."""
    p = params_for(db, obj.name, obj.category)
    arm = choose_arm(obj)
    if p.get("preset", "power") == "pinch" and obj.size.z <= FLAT_MAX_HEIGHT:
        # The upright thumb pad is 7 cm above the open index. A taller object keeps the upright rim grasp:
        # the leaned pre-grasp over a 13 cm box had no IK (197 mm).
        flat = plan_flat_pinches(obj, scene, p, arm)
        if flat:
            return flat
    if p.get("preset", "power") == "pinch":
        p = dict(p)  # the thumb pad is 7 cm above the open index; keep that pad below the top
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
