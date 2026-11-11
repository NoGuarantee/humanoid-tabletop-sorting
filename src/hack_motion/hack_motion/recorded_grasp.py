"""Named object profiles shared by the existing Hand action and sorting skills."""
import json
import math
from importlib.resources import files


RHINO = json.loads(files('hack_motion').joinpath('profiles/rhino_e71.json').read_text())
RHINO.update(key='rhino_e71', family='rhino', lift_m=.12)
BOXES = json.loads(files('hack_motion').joinpath('profiles/boxes.json').read_text())['profiles']
PROFILES = {p['key']: p for p in [RHINO, *BOXES]}
if len(PROFILES) != 1 + len(BOXES):
    raise ValueError('duplicate recorded profile')
PRESHAPE = 'rhino_e71_preshape'
CLOSE = 'rhino_e71_close'
HAND_TARGETS, PRESETS = {}, {}
for key, profile in PROFILES.items():
    side = profile['hand']
    if side not in ('left', 'right') or not 0 < profile['effort_limit'] <= 1:
        raise ValueError('invalid profile hand or force')
    stages = dict(preshape=profile['preshape'][side], close=profile['closure']['positions'][side])
    if 'postlift' in profile:
        stages['postlift'] = profile['postlift']['positions'][side]
    if 'release' in profile:
        stages['release'] = [1.] * 6
    for stage, target in stages.items():
        if len(target) != 6 or not all(math.isfinite(v) and 0 <= v <= 1 for v in target):
            raise ValueError('invalid six-actuator target')
        preset = key + '_' + stage
        HAND_TARGETS[preset], PRESETS[preset] = target, (profile, stage)


def profile_for_preset(preset):
    return PRESETS[preset][0]


def hand_steps(preset):
    """Six actuator targets and sim-time dwell, including optional staged release."""
    profile, stage = PRESETS[preset]
    side = profile['hand']
    if stage == 'preshape':
        return [([1.] * 6, .5), (profile['preshape'][side], .5)]
    if stage == 'close':
        closure, start = profile['closure'], profile['preshape'][side]
    elif stage == 'postlift':
        closure, start = profile['postlift'], profile['closure']['positions'][side]
    else:
        closure = profile['release']
        start = profile.get('postlift', profile['closure'])['positions'][side]
    end = HAND_TARGETS[preset]
    intervals = closure.get('progress_intervals', [[0., 1.]] * 6)
    if (len(intervals) != 6 or any(not 0 <= a < b <= 1 for a, b in intervals)
            or not 1 <= closure['steps'] <= 200 or not 0 < closure['duration_s'] <= 10
            or not 0 <= closure['settle_s'] <= 3):
        raise ValueError('invalid recorded schedule')
    steps = []
    for index in range(1, closure['steps'] + 1):
        progress = index / closure['steps']
        positions = []
        for a, b, (begin, finish) in zip(start, end, intervals):
            alpha = min(1., max(0., (progress - begin) / (finish - begin)))
            positions.append((1 - alpha) * a + alpha * b)
        steps.append((positions, closure['duration_s'] / closure['steps']))
    steps.append((end, closure['settle_s']))
    return steps
