"""Validate one fresh, explicitly selected contact object for a recorded descent."""
import math
import time


def measured_contact(goal, objects, read_now, read_tcp, carrying=False, cancelled=lambda: False):
    """Sample TF before the clock; allow their independent subscriptions to catch up.

    No motor commands or relaxed age limits. A fresh TF may arrive one tick
    ahead of /clock; keep that sample while waiting instead of chasing the
    next future TF. Missing/stale TF gets a new sample within the wall budget.
    """
    deadline = time.monotonic() + .5
    tcp = read_tcp()
    while True:
        if cancelled():
            raise ValueError('recorded_contact_cancelled')
        now = read_now()
        try:
            return contact_object(goal, objects, now, tcp, carrying)
        except ValueError as exc:
            if str(exc).split(':', 1)[0] not in ('stale_recorded_contact_tcp', 'recorded_contact_requires_measured_tcp'):
                raise
            if time.monotonic() >= deadline:
                raise
        time.sleep(.01)
        stamp = tcp.header.stamp.sec + tcp.header.stamp.nanosec/1e9 if tcp is not None else None
        if stamp is None or stamp <= now:
            tcp = read_tcp()


def contact_object(goal, objects, now, tcp, carrying=False):
    identity = getattr(goal, 'contact_object', '')
    if not identity:
        return ''
    if goal.mode != 'linear_fixed' or carrying:
        raise ValueError('recorded_contact_requires_unloaded_descent')
    stamp = goal.contact_stamp_ns
    if type(stamp) is not int or not 0 <= now-stamp/1e9 <= 1.75:
        raise ValueError('stale_recorded_contact')
    point = tuple(goal.contact_center)
    if len(point) != 3 or not all(math.isfinite(v) for v in point):
        raise ValueError('invalid_recorded_contact_center')
    matches = [obj for obj in objects if obj[0] == identity]
    if len(matches) != 1:
        raise ValueError('recorded_contact_object_missing_or_ambiguous')
    obj = matches[0]
    if not all(math.isfinite(v) for v in obj[1:8]) or min(obj[5:8]) <= 0:
        raise ValueError('invalid_recorded_contact_bounds')
    c, s = math.cos(obj[4]), math.sin(obj[4])
    dx, dy, dz = (point[k]-obj[k+1] for k in range(3))
    local = (c*dx+s*dy, -s*dx+c*dy, dz)
    if any(abs(v) > size/2+.02 for v, size in zip(local, obj[5:8])):
        raise ValueError('recorded_contact_object_moved_or_reused')
    if tcp is None:
        raise ValueError('recorded_contact_requires_measured_tcp')
    measured_stamp = tcp.header.stamp.sec + tcp.header.stamp.nanosec/1e9
    if not 0 <= now-measured_stamp <= .5:
        raise ValueError(f'stale_recorded_contact_tcp: age={now-measured_stamp:.9f}s now={now:.9f} stamp={measured_stamp:.9f}')
    target, start = goal.target.position, tcp.transform.translation
    end = (target.x, target.y, target.z)
    before = (start.x, start.y, start.z)
    if (not all(math.isfinite(v) for v in (*end, *before)) or end[2] > before[2]-.01
            or math.dist(end, before) > .20 or math.dist(end, point) > .30):
        raise ValueError('recorded_contact_outside_descent')
    return identity
