"""Command construction only; the FSM owns the composite skill sequence.

Tube insertion/reorientation deliberately have no fabricated geometric implementation.
"""
from .context import State


class Skills:
    def __init__(self, db):
        self.db = db
        self.grasp = None
        self.place = None

    def command(self, effect, scene, attempt=1):
        from hack_interfaces.action import Hand, MoveJoints, MoveTcp
        from .grasp_planner import plan_grasps, plan_place
        phase = effect.phase
        if phase == State.LOOK_TABLE:
            return 'joints', MoveJoints.Goal(
                joints=['head_pitch_joint', 'head_yaw_joint', 'body_yaw_joint'],
                positions=[0.4, 0.0, 0.0])
        if phase == State.OPEN:
            obj = next(o for o in scene.objects if (o.name or o.track_id) == effect.target.id)
            candidates = plan_grasps(obj, scene, self.db)
            if not candidates:
                raise ValueError('no_grasp_candidates')
            self.grasp = candidates[min(attempt - 1, len(candidates) - 1)]
            return 'hand', Hand.Goal(hand=self.grasp.arm, preset='open')
        if self.grasp is None:
            raise ValueError('missing_grasp_plan')
        g = self.grasp
        if phase in (State.CLOSE, State.RELEASE):
            return 'hand', Hand.Goal(hand=g.arm, preset=g.hand_preset if phase == State.CLOSE else 'open',
                                     force=float(g.force if phase == State.CLOSE else 0.0))
        if phase == State.TRANSFER:
            container = next(x for x in scene.containers if x.id == effect.destination and x.kind == 'tray')
            self.place = plan_place(g.arm, container, self.db)
        poses = {State.APPROACH: g.pre_grasp, State.DESCEND: g.grasp, State.LIFT: g.lift}
        if self.place:
            poses.update({State.TRANSFER: self.place.approach, State.LOWER: self.place.release,
                          State.RETREAT: self.place.retreat})
        if phase not in poses:
            raise NotImplementedError('skill unavailable: ' + phase.value)
        return 'tcp', MoveTcp.Goal(arm=g.arm, target=poses[phase],
                                  mode='direct' if phase in (State.APPROACH, State.TRANSFER) else 'linear')
