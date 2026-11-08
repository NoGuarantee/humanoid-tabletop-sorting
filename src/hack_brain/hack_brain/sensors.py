"""Timestamped measurements; never turn action success into physical evidence."""
from collections import deque
import math
import threading

# Finger reading spread over 0.3 s that still counts as a hand at rest, for the hold reading and for the stop
# after a hand op. A thumb squeezing at full effort jitters: 0.018 on the sink, 0.08 pinching the cartridge,
# where 0.04 never saw the hand settle and the close ended the mission as an unconfirmed stop (eval 2026-11-08).
# A closing finger crosses 0..1 in about a second, far more than this in 0.3 s.
HAND_STILL = 0.12


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

    def still_report(self, key, now, since, names=None, duration=0.3, tolerance=0.004):
        """Why window() finds no rest: for the log when a stop stays unconfirmed."""
        with self.lock:
            samples = list(self.samples[key])
        rows = [row for row in samples if row[0] >= since and now - 0.6 <= row[0] <= now]
        if not rows:
            last = f"{now - samples[-1][0]:.2f} s old" if samples else "none"
            return f"{key}: no samples in the window, last {last}"
        rows = [row for row in rows if row[0] >= rows[-1][0] - duration - 0.1]
        names = tuple(names or rows[-1][1])
        gaps = max((b[0] - a[0] for a, b in zip(rows, rows[1:])), default=0.)
        moving = sorted(((max(r[1][n] for r in rows) - min(r[1][n] for r in rows), n) for n in names
                         if all(n in r[1] for r in rows)), reverse=True)
        moving = [f"{n} {d:.4f}" for d, n in moving if d > tolerance][:6]
        return (f"{key}: {len(rows)} rows over {rows[-1][0] - rows[0][0]:.2f} s, last {now - rows[-1][0]:.2f} s old, "
                f"max gap {gaps:.2f} s, over {tolerance}: {', '.join(moving) or 'none'}")

    def hand(self, side, now, since, target=None):
        row = self.window(side, now, since, tuple(str(i) for i in range(1,7)), tolerance=HAND_STILL)
        if row is None:
            return None
        t, values, msg = row
        pos = [values[str(i)] for i in range(1,7)]
        # A finger pressed against an object or the table reads a little past open (1.03 on a held cartridge,
        # 1.34 with a pad on the table, 1.54 on the little finger with pinch_flat on the bowl rim), and a finger
        # curled hard past an edge a little below closed (-0.13 on the middle finger pinching the cartridge).
        # Both once read as no hand: physical_unknown ended the mission (eval 2026-11-08). Clamp that;
        # a reading far outside 0..1 is not a hand.
        if any(not -.3 <= p <= 1.8 for p in pos):
            return None
        pos = [min(max(p, 0.), 1.) for p in pos]
        # Five closing fingers open hold nothing. Thumb rotation (6) is not one: pinch_ready keeps it at 0.1,
        # and a hand left there by a failed approach read UNKNOWN until the mission ended (eval 2026-11-08).
        if all(p >= 0.95 for p in pos[:5]):
            return t, 'EMPTY'
        if target is None:
            return None
        if len(msg.effort) != len(msg.name) or not all(math.isfinite(x) for x in msg.effort):
            return None
        # Require sustained contact, not a one-frame force spike: a closing finger stopped short of its
        # target and pushing. One finger is enough. The pinch that carried the cartridge on the bench read
        # (1.00 1.00 1.00 1.03 0.25 0.10) for target (1 1 1 0.3 0.4 0.1): the index stopped at the object
        # and the thumb went past its target, so "thumb and another finger both stopped" never held.
        # VERIFY_LIFT still needs the TCP risen with the contact unbroken, so a finger on the table fails there.
        with self.lock:
            rows = [row for row in self.samples[side] if t-.3 <= row[0] <= t]
        contact = set(range(5))
        for _, positions, frame in rows:
            if len(frame.effort) != len(frame.name) or not all(math.isfinite(x) for x in frame.effort):
                return None
            effort = dict(zip(frame.name, frame.effort))
            contact &= {i for i in range(5) if target[i] < .9
                        and min(positions.get(str(i+1), -1), 1.) > target[i]+.05
                        and effort.get(str(i+1), 0) > .1}
        if len(rows) >= 3 and contact:
            return t, 'HOLDING'
        # Closed to the target or past it (the thumb overshoots) with nothing in the way.
        if all(p <= g + .05 for p, g in zip(pos, target) if g < .9):
            return t, 'EMPTY'
        return None
