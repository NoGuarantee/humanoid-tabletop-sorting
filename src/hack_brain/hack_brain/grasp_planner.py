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
    leaned: bool = False  # palm rolled off vertical: straight-line descent only, carried by the point between the pads


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


def grasp_xy(obj, tcp_yaw, preset, arm, p=None):
    """Where `tcp_at` aims so the object sits between the pads. Objects up to FLAT_MAX_HEIGHT use `plan_flat_pinches`.

    grasp_db `ahead` and `across` (m, right-hand TCP frame) move the aim; `tools/pick_one.py --sweep` searches them.

    The palm faces TCP +y on the right hand and -y on the left, so the object has to lie on that side of the
    axis. Choosing the side by distance from the pelvis put a right-hand object behind the back of the hand.
    """
    x, y = obj.center.x, obj.center.y
    if preset != "pinch":
        return x, y
    # The index-to-thumb line (FLAT_INDEX, FLAT_THUMB) runs 3 cm ahead of the axis, between 2.2 and 9.1 cm across.
    # Aimed at the axis, that line crossed a long face 1 cm from its end and the pads turned the object.
    p = p or {}
    ahead = p.get("ahead", 0.5 * (FLAT_INDEX[0] + FLAT_THUMB[0]))
    across = p.get("across", 0.5 * (FLAT_INDEX[1] + FLAT_THUMB[1]))
    if arm == "left":
        across = -across
    c, s = math.cos(tcp_yaw), math.sin(tcp_yaw)
    return x - ahead * c + across * s, y - ahead * s - across * c


# Ready fingertips in the right-hand TCP frame: fingers open, thumb rotated to actuator 6 = 0.1, thumb bend open.
# The left hand mirrors y. At 45° the two pads are level and 9.8 cm apart; the index closes onto the thumb.
# 55° left the thumb 1.7 cm lower, and on a 3 cm dish that pad was on the table (bend reading 1.34, past open).
# A fully sideways hand (wrist at the table, z ≈ -0.14) has no IK.
FLAT_INDEX = (0.0294, 0.0220, -0.2088)
FLAT_THUMB = (0.0365, 0.0913, -0.1401)
UPRIGHT_APERTURE = FLAT_THUMB[1] - FLAT_INDEX[1]  # across TCP y, hand vertical
# The other open tips, beside the index. Spread is along TCP x. A yaw that points that
# spread into the dish sets the middle and ring on the top while the index closes in the
# air. From the URDF knuckle origins, plus the index tip's offset from FLAT_KNUCKLE.
FLAT_MIDDLE = (0.0119, 0.0220, -0.2090)
FLAT_RING = (-0.0094, 0.0220, -0.2082)
FLAT_LITTLE = (-0.0285, 0.0220, -0.2064)
# right_index_1 origin in that same frame. It sits behind the pads; the pinch is placed on the far side of it.
FLAT_KNUCKLE = (0.020493, -0.000285, -0.136914)
# 45° put the pads level, and the finger then crosses the outer face above the rim: the close
# stops on that edge. 32° hangs the finger nearer to vertical, so the tip, not the knuckle,
# is what meets the wall.
FLAT_ROLL = math.radians(32)
FLAT_APERTURE = 0.098  # tip-to-tip distance of the ready hand
# Open index tip outside the outer face. A 4 cm gap closed to the command and touched nothing.
# Closer than 1.2 cm, the knuckle is already on the rim and the close stops before the tip moves.
INDEX_NEAR = (0.012, 0.028)
# Thumb pad inside the opening, past the 12 mm rim. At 32° it is 2.2 cm above the index, so
# the 2.6 cm index height leaves the thumb just under the rim.
THUMB_IN = (-0.055, -0.012)
# Index tip at actuator 4 = 0.48 minus the open tip, right-hand TCP frame. The finger chain
# below reproduces this; the placement uses the whole path (`_face_under`), not this one point.
# Past 0.48 the tip climbs back out, so the close aims at 0.55 and the wall is what stops it.
FLAT_CURL = (-0.0051, 0.0463, 0.0631)
# Thumb links at pinch_ready, same frame. A point at or below the top has to be in the opening.
FLAT_THUMB_LINKS = ((0.0224, 0.0666, -0.1013), (0.0263, 0.0811, -0.1183), FLAT_THUMB)
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


# Open index tip below the rim. At 32° the thumb is 2.2 cm higher, so 2.6 cm puts the thumb
# just under the rim. 1.2 cm (thumb above the rim) stops the index at 0.86 and it slips on the
# lift. 3.2 cm, with the tip 4 cm outside, closed to the command and touched nothing.
INDEX_UNDER_RIM = 0.026
INDEX_CLEAR_TABLE = 0.018


def _flat_pad_z(obj, scene):
    """Height of the open index tip: in the band where the close meets the dish, and off the table."""
    local = obj.top_z - obj.size.z if obj.size.z > 0 else scene.table_z
    floor = max(scene.table_z, local)
    index_z = obj.top_z - INDEX_UNDER_RIM
    if index_z < floor + INDEX_CLEAR_TABLE:
        return None
    return index_z


def _finger_on_top(obj, index_xy, _pad_z, i_w, k_w, hand):
    """The open index would be set down on the top face by a straight descent.

    `index_xy` is the pad relative to the centre, `hand` points out of the dish. The pad can finish
    beside the wall while the finger behind it is still above the top. The descent lowers that part
    through the top height at the same xy, so the whole finger counts, not only the part that ends
    below the top. The finger is about 1 cm thick on the inner side of that line.
    """
    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        p = (index_xy[0] + a * (k_w[0] - i_w[0]), index_xy[1] + a * (k_w[1] - i_w[1]))
        belly = (p[0] - 0.008 * hand[0], p[1] - 0.008 * hand[1])
        # Under 12 mm into the box is the rim. Deeper, the finger comes down on the opening.
        if _box_clear(obj, p) < -0.012 or _box_clear(obj, belly) < -0.012:
            return True
    return False


def _beside_on_dish(obj, index_xy, i_w, rot, sign):
    """Middle, ring or little would come down on the opening. The index is the finger that closes.

    The tip can sit outside the box while the finger behind it lies 2 cm into the opening.
    Each pair is the knuckle and the open tip in the right-hand TCP frame.
    """
    pairs = (
        ((0.00295, -0.000285, -0.1371), FLAT_MIDDLE),
        ((-0.01827, -0.000285, -0.1363), FLAT_RING),
        ((-0.03742, -0.000285, -0.1345), FLAT_LITTLE),
    )
    for knuckle, tip in pairs:
        kw = matvec(rot, (knuckle[0], sign * knuckle[1], knuckle[2]))
        tw = matvec(rot, (tip[0], sign * tip[1], tip[2]))
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            w = tuple((1 - a) * kw[j] + a * tw[j] for j in range(3))
            p = (index_xy[0] + w[0] - i_w[0], index_xy[1] + w[1] - i_w[1])
            if _box_clear(obj, p) < -0.012:
                return True
    return False


# right_index chain from the URDF. Positive rotation is the close: the tip at actuator 0.48
# (52% of the way shut) is FLAT_INDEX + FLAT_CURL. The sim commands the opposite sign because
# its joint axis is flipped; the finger moves the same way.
_INDEX_AXIS = (0.0, -0.069756, -0.997564)
_INDEX_TIP = (0.0142, 0.0411, -0.0053)  # in right_index_2, the open tip FLAT_INDEX
_INDEX_TRAVEL = 1.333  # rad from open to shut
_INDEX_MIMIC = 1.1169
# Closing fractions at which the tip is checked. 0.20 is actuator 0.80, where a tip that
# started 3 cm outside reaches the face while it is still at the open height.
_CURL_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.52)


def _rx(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return (
        (c + x * x * k, x * y * k - z * s, x * z * k + y * s),
        (y * x * k + z * s, c + y * y * k, y * z * k - x * s),
        (z * x * k - y * s, z * y * k + x * s, c + z * z * k),
    )


def _index_tip_tcp(closing):
    """Right-hand TCP frame position of the index tip, `closing` from 0 (open) to 1 (shut)."""
    q1 = closing * _INDEX_TRAVEL
    q2 = _INDEX_MIMIC * q1
    p = matvec(_rx((0.0, 0.0, -1.0), q2), _INDEX_TIP)
    p = tuple(p[i] + (0.008099, 0.031095, -0.001)[i] for i in range(3))  # index_2 origin, in index_1
    p = matvec(_rx(_INDEX_AXIS, q1), p)
    p = matvec(rpy_matrix(1.5359, 0.0, 0.0), p)
    p = tuple(p[i] + (-0.000285, 0.020275, 0.136946)[i] for i in range(3))  # index_1 origin, in the palm
    return matvec(rpy_matrix(3.14, 0.0, 1.570796), p)  # palm to TCP


_TIP_OPEN = _index_tip_tcp(0.0)
_CURL_DELTAS = tuple(
    tuple(_index_tip_tcp(f)[i] - _TIP_OPEN[i] for i in range(3)) for f in _CURL_FRACTIONS)


def _face_under(obj, index_xy, rot, sign, pad_z):
    """How far under the rim the index tip is when it first reaches the outer face.

    None when the tip climbs to the top edge before it has crossed that face: the close then
    stops on the edge and the rest of the curl, on the lift, happens in the air.
    """
    for delta in _CURL_DELTAS:
        local = (delta[0], sign * delta[1], delta[2])
        dw = matvec(rot, local)
        xy = (index_xy[0] + dw[0], index_xy[1] + dw[1])
        z = pad_z + dw[2]
        if _box_clear(obj, xy) > 0.0:
            continue  # still outside
        if z > obj.top_z - 0.012:
            return None
        return obj.top_z - z
    return None


def _thumb_in_opening(obj, index_xy, rot, sign, pad_z):
    """Thumb links that reach the dish height sit in the opening, not on the rim."""
    index = (FLAT_INDEX[0], sign * FLAT_INDEX[1], FLAT_INDEX[2])
    for pt in FLAT_THUMB_LINKS:
        local = (pt[0] - index[0], sign * pt[1] - index[1], pt[2] - index[2])
        w = matvec(rot, local)
        if pad_z + w[2] <= obj.top_z and _box_clear(obj, (index_xy[0] + w[0], index_xy[1] + w[1])) > -0.012:
            return False
    return True


def plan_flat_pinches(obj, scene, p, arm):
    """Pinch a flat object with the palm leaned 32° and the thumb already in the opening.

    Upright, the thumb pad is 7 cm above the open index, so on a short dish the index curls onto the
    top. At 45° the pads are level and the finger crosses the outer face above the rim, so the close
    stops on that edge. At 32° the tip leads. It starts within 3 cm of the wall and 2.6 cm under the
    rim, which puts the thumb just under the rim. A 4 cm gap closes in the air. The wrist stays
    near z = 0, which IK can reach.
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
                if not (INDEX_NEAR[0] <= near <= INDEX_NEAR[1] and THUMB_IN[0] <= far <= THUMB_IN[1]):
                    continue
                tcp_x = obj.center.x + idx[0] - i_w[0]
                tcp_y = obj.center.y + idx[1] - i_w[1]
                # x 0.26 stalled 15 cm short of the object.
                if tcp_x < 0.30:
                    continue
                if _finger_on_top(obj, idx, pad_z, i_w, k_w, hand):
                    continue
                if _beside_on_dish(obj, idx, i_w, rot, sign):
                    continue
                if not _thumb_in_opening(obj, idx, rot, sign, pad_z):
                    continue
                depth = _face_under(obj, idx, rot, sign, pad_z)
                if depth is None:
                    continue
                # Wrist just in front of this object, on the arm's side of it. A fixed
                # (0.32, -0.10) pulled the forearm across the centre tray once the dish
                # sat in that tray's corner.
                side = -1.0 if arm == "right" else 1.0
                reach = -math.hypot(tcp_x - (obj.center.x - 0.05), tcp_y - (obj.center.y + side * 0.04))
                rank = depth + 0.4 * reach
                if best is None or rank > best[0]:
                    best = (rank, depth, reach, idx, i_w, yaw)
        if best is not None:
            found.append(best)
    def _rank(item):
        _inner, depth, reach, _idx, _iw, yaw = item
        # The nearer angle is the smaller wrist turn from the parked hand.
        dy = abs((yaw - base_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        return -(depth - 0.25 * dy + 0.4 * reach)

    found.sort(key=_rank)
    out = []
    yaws = []
    for _inner, depth, _reach, index_xy, i_w, yaw in found:
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
            score=depth,
            approach_preset="pinch_ready",
            leaned=True,
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
        p = dict(p)  # the thumb pad is 6.9 cm above the open index; keep that pad 2 cm below the top
        p["z_offset"] = -min(0.6 * obj.size.z, FLAT_THUMB[2] - FLAT_INDEX[2] + 0.02)
    z = fingertip_z(obj, scene, p)  # tips, not the TCP: tcp_at lifts the palm by FINGERTIP
    base_yaw = pinch_yaw(obj.yaw, p.get("align", "short_axis"))
    preset = p.get("preset", "power")
    out = []
    for i, dyaw in enumerate((0.0, math.pi / 2)):  # second candidate: rotated hand, used on a retry
        yaw = base_yaw + dyaw
        # A retry across a side wider than the pads would land the thumb on top of the object.
        if preset == "pinch" and i > 0 and span_across(obj, yaw) > UPRIGHT_APERTURE:
            continue
        q = top_down(yaw)
        x, y = grasp_xy(obj, yaw, preset, arm, p)
        out.append(GraspCandidate(
            arm=arm,
            pre_grasp=tcp_at(x, y, z + p.get("pre_height", 0.10), q),
            grasp=tcp_at(x, y, z, q),
            lift=tcp_at(x, y, z + p.get("lift_height", 0.15), q),
            hand_preset=preset,
            force=p.get("force", 0.0),
            score=1.0 - 0.5 * i,
            # The thumb swings across on the close. Set it before the descent so it does not sweep the object.
            approach_preset="pinch_ready" if preset == "pinch" else "open",
        ))
    return out


def place_point(container, inset, margin):
    """Drop point in the tray opening, `inset` from the centre towards the robot (pelvis origin).

    The side trays lie at the edge of the reach: above tray_left (y 0.69) the left arm got only to y 0.63-0.65 and
    TRANSFER failed with the object in the hand (eval --layouts 3, 2026-11-09). The shift stays `margin` inside the
    footprint `size` turned by the tray yaw; an unknown size gives the centre.
    """
    c = container.pose.position
    d = math.hypot(c.x, c.y)
    if d < 1e-6 or inset <= 0.0:
        return c.x, c.y
    dx, dy = -c.x / d * inset, -c.y / d * inset
    o = container.pose.orientation
    yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))
    cy, sy = math.cos(yaw), math.sin(yaw)
    lx, ly = cy * dx + sy * dy, -sy * dx + cy * dy  # into the tray frame
    hx = max(0.0, container.size.x / 2.0 - margin)
    hy = max(0.0, container.size.y / 2.0 - margin)
    lx, ly = max(-hx, min(hx, lx)), max(-hy, min(hy, ly))
    return c.x + cy * lx - sy * ly, c.y + sy * lx + cy * ly


def plan_place(arm, container, db):
    """Drop above a tray, a little towards the robot from its centre. Tubes into slots: TODO (milestone M4)."""
    p = db.get("place", {})
    c = container.pose.position
    x, y = place_point(container, p.get("inset", 0.0), p.get("rim_margin", 0.06))
    q = top_down(0.0)
    above = c.z + p.get("approach_height", 0.20)  # fingertips above the opening; tcp_at raises the palm
    return PlacePlan(arm=arm,
                     approach=tcp_at(x, y, above, q),
                     release=tcp_at(x, y, c.z + p.get("release_height", 0.10), q),
                     retreat=tcp_at(x, y, above, q),
                     waist_yaw=0.0)
