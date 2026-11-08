"""Evidence validation, deliberately no uncalibrated force heuristic."""
import math
from .context import Evidence, Hand


def fresh(stamp, now, max_age, since=None):
    return (now is not None and math.isfinite(stamp) and math.isfinite(now)
            and 0 <= now - stamp <= max_age and (since is None or stamp >= since))


def evidence(c, value, since):
    if not isinstance(value, Evidence) or not value.source:
        return None
    if not fresh(value.stamp, c.sim, c.config.freshness, since):
        return None
    if value.hand == Hand.HOLDING and (c.target is None or value.object_id != c.target.id):
        return None
    return value


def placement(c, value):
    return (value.hand == Hand.EMPTY and value.placed and value.object_id == c.target.id
            and value.destination == c.destination and value.slot == c.slot
            and (c.spec.kind == 'sort' or value.upright))
