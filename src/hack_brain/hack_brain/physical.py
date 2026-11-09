"""Sensor-based sorting evidence. Thresholds are tunable, not ground-truth labels.

Holding requires a closing finger held short of its target with effort (sensors.hand). Lift additionally requires measured TCP
rise with uninterrupted contact. Placement requires a unique named RGB-D detection
inside the observed tray after release and retreat. Occlusion stays UNKNOWN.
"""
import math
from .context import Evidence, Hand, State
from .sensors import stamp


def in_tray(obj, tray):
    p, q = tray.pose.position, tray.pose.orientation
    yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
    dx, dy = obj.center.x-p.x, obj.center.y-p.y
    x, y = math.cos(yaw)*dx+math.sin(yaw)*dy, -math.sin(yaw)*dx+math.cos(yaw)*dy
    values = (x,y,obj.center.z,obj.size.x,obj.size.y,obj.size.z,tray.size.x,tray.size.y,p.z)
    if not all(math.isfinite(v) for v in values) or min(obj.size.x,obj.size.y,obj.size.z,tray.size.x,tray.size.y) <= 0:
        return False
    # Bounding rectangle rotated into the tray frame, with margin from the rim.
    angle = obj.yaw-yaw
    ex = (abs(math.cos(angle))*obj.size.x + abs(math.sin(angle))*obj.size.y)/2
    ey = (abs(math.sin(angle))*obj.size.x + abs(math.cos(angle))*obj.size.y)/2
    return (abs(x)+ex < tray.size.x/2-.005 and abs(y)+ey < tray.size.y/2-.005
            and p.z-max(tray.size.z, .08) <= obj.center.z <= p.z+obj.size.z/2+.02)


class PhysicalVerifier:
    def __init__(self, measurements, tcp):
        self.measurements, self.tcp = measurements, tcp
        self.target_id = ''
        self.arm = None
        self.closure = None
        # Last closing target per arm. OPEN clears self.closure for the new grasp, but the other hand may
        # still be shut on nothing after a slipped attempt; VERIFY_EMPTY needs its target to read it empty.
        self.closures = {}
        self.baseline = None
        self.contact_seen = False
        self.lifted = False
        self.contact_lost = False
        self.unsure_since = None  # sim time the hand monitor stopped reading HOLDING (Clients.poll)

    def closing(self, target, arm, closure, now):
        self.target_id, self.arm, self.closure = target.id, arm, tuple(closure)
        self.closures[arm] = self.closure
        self.baseline = self.tcp(arm, now)
        self.contact_seen = self.lifted = self.contact_lost = False
        self.unsure_since = None

    def read(self, c, scene=None):
        since = c.active.sim_start
        if c.state == State.VERIFY_EMPTY:
            # A hand closed to its target with nothing in it is empty too: the right hand left shut by a slipped
            # cartridge read UNKNOWN here after the left placed it, and ended the mission with 261 s left
            # (eval 2026-11-09).
            hands = [self.measurements.hand(side,c.sim,since,self.closures.get(side)) for side in ('left','right')]
            if all(h is not None and h[1] == 'EMPTY' for h in hands):
                return Evidence(min(h[0] for h in hands), Hand.EMPTY, source='measured_open_hands')
            return None
        if c.target is None or self.arm is None:
            return None
        sample = self.measurements.hand(self.arm,c.sim,since,self.closure)
        if sample is None:
            return None
        t, hand = sample
        if hand == 'HOLDING' and self.target_id == c.target.id and not self.contact_lost:
            self.contact_seen = True
            current = self.tcp(self.arm,c.sim)
            if self.baseline is None and c.state == State.VERIFY_GRASP:
                # The arm has not moved since CLOSE: a pose missed then is still the grasp height.
                self.baseline = current
            if self.baseline and current and current[0] >= since:
                if current[1][2] - self.baseline[1][2] >= .05:
                    self.lifted = True
                t = min(t,current[0])
            elif c.state == State.VERIFY_LIFT and not self.lifted:
                return None  # no TCP pose after the lift yet: wait instead of reporting it not lifted
            return Evidence(t,Hand.HOLDING,c.target.id,lifted=self.lifted,source='opposed_finger_contact_and_tcp')
        if hand != 'EMPTY':
            return None
        if self.contact_seen:
            self.contact_lost = True
        placed = False
        if c.state != State.VERIFY_PLACE:
            scene = None
        elif scene is not None:
            st = stamp(scene)
            # A frame taken after this check started is after the arm left the view, however long perception
            # took: one 2.1 s old was dropped and read the open, empty hand UNKNOWN, which ended the mission after
            # a sorted sink with 308 s left (eval --layouts 3, 2026-11-09). An earlier frame shows nothing about
            # the placement; the hand still reads, so the object counts released, not placed.
            if scene.header.frame_id != 'pelvis' or not since <= st <= c.sim:
                scene = None
        if scene is not None:
            # track_id is only stable before contact; never rename a nearby detection.
            matches = [o for o in scene.objects if c.target.name and o.name == c.target.name
                       and not o.partially_visible and o.confidence >= c.config.min_confidence]
            trays = [x for x in scene.containers if x.id == c.destination and x.kind == 'tray']
            placed = len(matches)==1 and len(trays)==1 and in_tray(matches[0],trays[0])
            t = min(t,st)
        return Evidence(t,Hand.EMPTY,c.target.id,placed=placed,destination=c.destination,
                        slot=c.slot,source='measured_hand_and_rgbd' if scene else 'measured_hand')
