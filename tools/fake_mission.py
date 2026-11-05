#!/usr/bin/env python3
"""Exercise the real mission reducer with explicitly synthetic evidence, no ROS."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src/hack_brain'), str(ROOT / 'tests')]
from fakes import Fake, tubes_spec
from hack_brain.context import Object, State


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['sort', 'easy', 'standard', 'color', 'lying'], default='sort')
    args = parser.parse_args()
    fake = Fake() if args.variant == 'sort' else Fake(
        tubes_spec(args.variant), [Object('tube', 'falcon50', name='tube', cap='red', lying=args.variant == 'lying')])
    while fake.c.state != State.DONE:
        c = fake.c
        print(json.dumps({'state': c.state.value, 'operation': c.active.id, 'hand': c.hand.value,
                          'target': c.target.id if c.target else None, 'destination': c.destination, 'slot': c.slot}))
        fake.answer()
    print(json.dumps({'state': fake.c.state.value, 'reason': fake.c.reason,
                      'confirmed': sum(h.placed for h in fake.c.history.values()), 'evidence': 'FAKE ONLY'}))


if __name__ == '__main__':
    main()
