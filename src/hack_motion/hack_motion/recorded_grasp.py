"""Numeric E71 profile shared by the existing Hand action and pick client."""
import json
from importlib.resources import files


RHINO = json.loads(files('hack_motion').joinpath('profiles/rhino_e71.json').read_text())
PRESHAPE = 'rhino_e71_preshape'
CLOSE = 'rhino_e71_close'
HAND_TARGETS = {
    PRESHAPE: RHINO['preshape']['right'],
    CLOSE: RHINO['closure']['positions']['right'],
}


def hand_steps(preset):
    """Six simultaneous actuator targets and the following sim-time dwell."""
    if preset == PRESHAPE:
        return [(HAND_TARGETS[PRESHAPE], .5)]
    if preset != CLOSE:
        raise ValueError('unknown recorded preset')
    closure = RHINO['closure']
    start, end = HAND_TARGETS[PRESHAPE], HAND_TARGETS[CLOSE]
    steps = []
    for index in range(1, closure['steps'] + 1):
        alpha = index / closure['steps']
        positions = [(1 - alpha) * a + alpha * b for a, b in zip(start, end)]
        steps.append((positions, closure['duration_s'] / closure['steps']))
    steps.append((end, closure['settle_s']))
    return steps
