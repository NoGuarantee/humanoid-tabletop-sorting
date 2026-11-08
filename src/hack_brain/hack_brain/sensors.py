"""Timestamped measurements; never turn action success into physical evidence."""
from collections import deque
import math
import threading


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def joint_values(msg):
    if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
        return None
    values = dict(zip(msg.name, msg.position))
    return values if values and all(math.isfinite(v) for v in values.values()) else None


class Measurements:
    def __init__(self):
        self.lock = threading.RLock()
        self.samples = {key: deque(maxlen=100) for key in ('joints', 'left', 'right')}

    def update(self, key, msg):
        t, values = stamp(msg), joint_values(msg)
        with self.lock:
            samples = self.samples[key]
            if not math.isfinite(t) or values is None:
                samples.clear()
                return
            if samples and t <= samples[-1][0]:
                if t < samples[-1][0]:
                    samples.clear()
                return
            samples.append((t, values, msg))

    def window(self, key, now, since, names=None, duration=0.3, tolerance=0.004):
        with self.lock:
            samples = list(self.samples[key])
        rows = [row for row in samples if row[0] >= since and now - 0.6 <= row[0] <= now]
        if len(rows) < 3 or not 0 <= now - rows[-1][0] <= 0.2:
            return None
        # A recent continuous window is needed; missing/repeated frames cannot prove rest.
        rows = [row for row in rows if row[0] >= rows[-1][0] - duration - 0.1]
        if rows[-1][0] - rows[0][0] < duration or any(b[0]-a[0] > 0.2 for a,b in zip(rows,rows[1:])):
            return None
        names = tuple(names or rows[-1][1])
        if not names or any(any(n not in row[1] for n in names) for row in rows):
            return None
        if any(max(row[1][n] for row in rows)-min(row[1][n] for row in rows) > tolerance for n in names):
            return None
        return rows[-1]

    def hand(self, side, now, since, target=None):
        row = self.window(side, now, since, tuple(str(i) for i in range(1,7)), tolerance=0.015)
        if row is None:
            return None
        t, values, msg = row
        pos = [values[str(i)] for i in range(1,7)]
        if any(not 0 <= p <= 1 for p in pos):
            return None
        if all(p >= 0.95 for p in pos):
            return t, 'EMPTY'
        if target is None:
            return None
        if len(msg.effort) != len(msg.name) or not all(math.isfinite(x) for x in msg.effort):
            return None
        # Require sustained opposed contact, not a one-frame force spike.
        with self.lock:
            rows = [row for row in self.samples[side] if t-.3 <= row[0] <= t]
        contact = set(range(5))
        for _, positions, frame in rows:
            if len(frame.effort) != len(frame.name) or not all(math.isfinite(x) for x in frame.effort):
                return None
            effort = dict(zip(frame.name, frame.effort))
            contact &= {i for i in range(5) if target[i] < .9
                        and positions.get(str(i+1), -1) > target[i]+.05
                        and effort.get(str(i+1), 0) > .1}
        if len(rows) >= 3 and 4 in contact and any(i < 4 for i in contact):
            return t, 'HOLDING'
        if all(abs(p-g) < .04 for p,g in zip(pos,target)):
            return t, 'EMPTY'
        return None
