#!/usr/bin/env python3
"""Export selected, hash-checked full-cycle inputs as object-relative ROS trial profiles.

Offline only. World poses are removed here; no simulator state enters solution nodes.
The native controller and planner are not reproduced by copying the grasp geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = (
    ('brother_left_wide', 'brother_combo100_v187', 'b0_s20000', '3b619f1732d5dca723ebda4f4a83049a326a2f53c4af4ce38868baead90df8cc'),
    ('brother_left_compact', 'brother_combo100_v187', 'b0_s20002', 'b309d0ac0cae5b82b598558cfaeea2d910dcd3178a4b07ec1fdd0ecf3bc7daa2'),
    ('philips_right', 'philips100_v224', 'b4_s20000', '8d23cda0195278b3dcdc47e118bef27156bd5f73dda95ed80b34db3e13dbbace'),
    ('philips_right_turn180', 'philips100_v224', 'b4_s20007', '400c58831db70aafb60a482e7218af88b420e82db5c21680492174d17b73e3f6'),
    ('nescafe_right', 'coffee100_v215', 'b3_s20000', '0f5079c6e9864ae3d66fe83c2c6a8afe2fbadfa3c6d8f89ffeb86ab94b5f6e85'),
    ('nescafe_right_turn180', 'coffee100_v215', 'b3_s20006', '075b23af9050c4319f8a81659c50b243dc58cedbb31bc000839555d7ee3611c7'),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_matrix(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape != (7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:]) - 1) > 1e-5:
        raise ValueError('invalid recorded pose')
    w, x, y, z = p[3:] / np.linalg.norm(p[3:])
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = p[:3]
    return t


def convert(key, data, source):
    side = data['arm_side']
    if side not in ('left', 'right') or data.get('closure_schedule', 'simultaneous') != 'simultaneous':
        raise ValueError('unsupported source hand or closure schedule')
    frame = pose_matrix(data['object_pose_world_wxyz'])
    grasp = np.linalg.solve(frame, pose_matrix(data['target_tcp_pose_world_wxyz'])).tolist()
    pre = np.linalg.solve(frame, pose_matrix(data['pregrasp_tcp_pose_world_wxyz'])).tolist()
    commands = data['hand_command_affine_v1']
    profile = dict(key=key, id=key, family=key.split('_')[0], object_name=data['object_name'], hand=side,
        reference=dict(kind='rgbd_model_pose', model_name=data['object_name'], frame='canonical_rgbd_model', centroid_fallback=False),
        T_reference_pregrasp={side: pre}, T_reference_approach={side: grasp}, T_reference_tcp={side: grasp},
        preshape={side: commands['pregrasp_ratios']},
        closure=dict(positions={side: commands['closed_ratios']}, duration_s=data['closure_duration_s'],
                     settle_s=data.get('closure_settle_s', 0.), steps=40),
        release=dict(duration_s=2., settle_s=.5, steps=40,
                     progress_intervals=[[0., .75]]*5+[[.75, 1.]] if
                     data['box_transfer'].get('release_schedule') == 'flexion_then_opposition' else [[0., 1.]]*6),
        lift_m=data['requested_lift_m'], effort_limit=.6, source=source,
        adaptations=dict(ros_qualified=False, effort='ROS trial limit; no native force equivalence',
            preshape='ROS six actuator targets; native encoder-bias approach controller is not replayed',
            motion='main MoveTcp planning and tracking; native world and joint paths are not replayed',
            placement='main observed-tray placement; native measured seating is not reproduced',
            native_seat_before_release=bool(data['box_transfer'].get('seat_before_release'))))
    if 'postlift_closure_ratios' in data:
        profile['postlift'] = dict(positions={side: data['postlift_closure_ratios']},
                                  duration_s=data['postlift_closure_duration_s'], settle_s=.5, steps=40)
    if key.startswith('philips_'):
        # Source box_pick_plan._carry clears the observed rim by 15 cm.
        # Main's generic 20 cm makes this tilted TCP unreachable on seed 7.
        # Retain main's release clearance; native floor seating is not copied.
        profile['placement'] = dict(carry_clearance_m=.15)
        if key == 'philips_right':
            # ROS carry adaptation, not part of the native demonstration:
            # tip the held box toward the robot to avoid the ROS wrist stop.
            profile['placement']['pitch_preference_rad'] = -np.pi/6
            profile['adaptations']['placement'] = 'observed-tray placement with -30 deg pelvis-Y carry pitch; native measured seating is not reproduced'
    return profile


def build(project):
    base = project / 'outputs/box_pose_robustness_20261107'
    profiles, fixtures = [], []
    inputs = []
    for key, tag, case, expected in CASES:
        root = base / tag
        path = root / 'stages' / f'{tag}_{case}' / 'probe/full_arm_input.json'
        report = root / 'remote_progress_latest.json'
        row = next(r for r in json.loads(report.read_text())['results'] if r['id'] == case)
        inputs.append((key, path, report, row, expected, root / 'policy_library.json'))
    root = base / 'pick_evidence_001/results/pick_brother0_v3'
    report = root / 'independent_cycle_audit.json'
    inputs.append(('brother_right', root/'input.json', report, json.loads(report.read_text()),
                   '15ee1513ceaf6c8072a0f80b5dea143f4fdc62f59266caac9e1c74bee1e23398', None))
    for key, path, report, row, expected, policy in inputs:
        if digest(path) != expected or row['input_sha256'] != expected or not row['passed'] or not all(row['checks'].values()):
            raise ValueError('unverified full cycle: ' + key)
        data = json.loads(path.read_text())
        source = dict(input_path=str(path.relative_to(project)), input_sha256=expected,
            report_path=str(report.relative_to(project)), report_sha256=digest(report),
            canonical_trace_sha256=row['uncompressed_trace_sha256'], native_full_cycle_passed=True)
        if policy:
            source.update(policy_path=str(policy.relative_to(project)), policy_sha256=digest(policy))
        profile = convert(key, data, source)
        profiles.append(profile)
        fields = ('object_pose_world_wxyz', 'target_tcp_pose_world_wxyz', 'pregrasp_tcp_pose_world_wxyz',
                  'arm_side', 'closure_duration_s', 'closure_settle_s', 'postlift_closure_ratios',
                  'postlift_closure_duration_s')
        fixture = {k: data[k] for k in fields if k in data}
        fixture.update(key=key, source=source, pregrasp_ratios=data['hand_command_affine_v1']['pregrasp_ratios'],
                       closed_ratios=data['hand_command_affine_v1']['closed_ratios'])
        fixtures.append(fixture)
    return profiles, fixtures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    profiles, fixtures = build(args.project.resolve())
    repo = Path(__file__).resolve().parents[1]
    for path, data in [(repo/'src/hack_motion/hack_motion/profiles/boxes.json', dict(profiles=profiles)),
                       (repo/'tests/recorded_grasp_sources.json', fixtures)]:
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(f'Exported {len(profiles)} hash-checked profiles')
