"""Sensor-based sorting evidence. Thresholds are tunable, not ground-truth labels.

Holding requires opposed finger contact. Lift additionally requires measured TCP
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
        self.baseline = None
        self.contact_seen = False
        self.lifted = False
        self.contact_lost = False

    def closing(self, target, arm, closure, now):
        self.target_id, self.arm, self.closure = target.id, arm, tuple(closure)
        self.baseline = self.tcp(arm, now)
        self.contact_seen = self.lifted = self.contact_lost = False

    def read(self, c, scene=None):
        since = c.active.sim_start
        if c.state == State.VERIFY_EMPTY:
            hands = [self.measurements.hand(side,c.sim,since) for side in ('left','right')]
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
            if self.baseline and current and current[0] >= since:
                if current[1][2] - self.baseline[1][2] >= .05:
                    self.lifted = True
                t = min(t,current[0])
            return Evidence(t,Hand.HOLDING,c.target.id,lifted=self.lifted,source='opposed_finger_contact_and_tcp')
        if hand != 'EMPTY':
            return None
        if self.contact_seen:
            self.contact_lost = True
        placed = False
        if c.state == State.VERIFY_PLACE and scene is not None:
            st = stamp(scene)
            if scene.header.frame_id != 'pelvis' or not since <= st <= c.sim or c.sim-st > c.config.freshness:
                return None
            # track_id is only stable before contact; never rename a nearby detection.
            matches = [o for o in scene.objects if c.target.name and o.name == c.target.name
                       and not o.partially_visible and o.confidence >= c.config.min_confidence]
            trays = [x for x in scene.containers if x.id == c.destination and x.kind == 'tray']
            placed = len(matches)==1 and len(trays)==1 and in_tray(matches[0],trays[0])
            t = min(t,st)
        return Evidence(t,Hand.EMPTY,c.target.id,placed=placed,destination=c.destination,
                        slot=c.slot,source='measured_hand_and_rgbd' if scene else 'measured_hand')
