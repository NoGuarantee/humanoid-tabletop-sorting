"""Task-specific routing; shared execution policy."""
import math
from .context import History, Lift


def required_capabilities(spec):
    if spec.task not in ('sort2', 'sort3', 'tubes') or spec.n_items <= 0:
        return None
    caps = {'perception', 'pick', 'verification', 'transfer', 'release', 'retreat'}
    if spec.kind == 'sort':
        return caps | {'place'}
    rules = {'easy': 'any_slot', 'standard': 'by_type', 'color': 'by_cap_color', 'lying': 'by_type'}
    if rules.get(spec.variant) != spec.rule:
        return None
    return caps | {'insert', 'hood'} | ({'reorient'} if spec.variant == 'lying' else set())


def destinations(spec, obj):
    if spec.kind == 'sort':
        dest = spec.categories.get(obj.category)
        return [dest] if dest else []
    return spec.hood_racks_for(obj.category, obj.cap)


def destination_in_scene(c, scene, obj):
    for dest in destinations(c.spec, obj):
        for container in scene.containers:
            if container.id != dest:
                continue
            if c.spec.kind == 'sort' and container.kind == 'tray':
                return dest, None
            if c.spec.kind == 'tubes' and container.kind == 'rack':
                for slot in container.free_slots:
                    if (dest, slot) not in c.used_slots:
                        return dest, slot
    return None


def select(c, scene):
    candidates = []
    for obj in scene.objects:
        h = c.history.get(obj.id, History())
        if (not obj.id or obj.partial or not math.isfinite(obj.confidence) or obj.confidence < c.config.min_confidence or h.placed or h.released
                or h.attempts >= c.config.max_attempts or obj.name in c.config.skip_names
                or not destinations(c.spec, obj)):
            continue
        if c.spec.kind == 'sort' and destination_in_scene(c, scene, obj) is None:
            continue
        candidates.append(obj)
    # Known lifted objects deserve recovery priority. No geometric identity guesses.
    candidates.sort(key=lambda o: (c.history.get(o.id, History()).lift != Lift.LIFTED, o.id))
    return candidates[0] if candidates else None
