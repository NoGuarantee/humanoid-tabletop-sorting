"""Deterministic mission reducer. Effects are executed outside this module."""
from copy import deepcopy
import math

from .context import Context, Effect, Event, Hand, History, Lift, Operation, Scene, State
from .policies import destination_in_scene, required_capabilities, select
from .verification import evidence, fresh, placement

VERIFY = {State.VERIFY_EMPTY, State.VERIFY_GRASP, State.VERIFY_LIFT, State.VERIFY_REORIENT,
          State.VERIFY_HOLD, State.VERIFY_TRANSFER, State.VERIFY_PLACE, State.RECOVER}
NEXT = {State.OPEN: State.APPROACH, State.DESCEND: State.CLOSE,
        State.CLOSE: State.VERIFY_GRASP, State.LIFT: State.VERIFY_LIFT,
        State.REORIENT: State.VERIFY_REORIENT, State.TRANSFER: State.VERIFY_HOLD,
        State.LOWER: State.RELEASE, State.INSERT: State.RELEASE,
        State.RELEASE: State.VERIFY_PLACE, State.RETREAT: State.VERIFY_EMPTY}


def _enter(c, state, effects):
    assert c.active is None and not c.blocked and c.state != State.DONE
    c.state = state
    c.serial += 1
    c.active = Operation(c.serial, state, c.sim, c.wall)
    if state in (State.DESCEND, State.RELEASE):
        c.hand = Hand.UNKNOWN
        c.scene = None
        c.history[c.target.id].contacted = True
    if state == State.LIFT and c.history[c.target.id].lift != Lift.LIFTED:
        c.history[c.target.id].lift = Lift.UNKNOWN
    effects.append(Effect('RUN', c.serial, state, c.target, c.destination, c.slot))


def _done(c, effects, reason=None):
    c.reason = reason or c.reason
    c.state = State.DONE
    effects.append(Effect('DONE'))


def _stop(c, effects, reason):
    if c.state == State.CANCELLING:
        return
    c.reason = reason
    if c.active is None:
        _done(c, effects)
    else:
        c.state = State.CANCELLING
        c.cancel_wall = c.wall
        effects.append(Effect('CANCEL', c.active.id))


def _recover(c, effects, reason):
    c.scene = None
    c.recoveries += 1
    if c.recoveries > c.config.max_recoveries:
        _done(c, effects, 'retries_exhausted:' + reason)
    elif c.target is None:
        _enter(c, State.OBSERVE, effects)
    else:
        _enter(c, State.RECOVER, effects)


def _tick(c, e, effects):
    sim, wall = e.data
    if c.started_wall is None:
        c.started_wall = c.last_progress_wall = wall
    c.wall = wall
    if c.state == State.CANCELLING:
        if wall - c.cancel_wall >= c.config.cancel_timeout:
            c.blocked = True
            _done(c, effects, c.reason + ':stop_unconfirmed')
        return
    if sim is not None:
        if not math.isfinite(sim) or (c.sim is not None and sim < c.sim):
            _stop(c, effects, 'clock_reset')
            return
        if c.sim is None or sim > c.sim:
            c.last_progress_wall = wall
        c.sim = sim
    if wall - c.last_progress_wall >= c.config.watchdog:
        _stop(c, effects, 'clock_missing' if c.sim is None else 'clock_stalled')
        return
    if c.state == State.WAIT:
        if c.spec is None or c.sim is None:
            if wall - c.started_wall >= c.config.watchdog:
                _stop(c, effects, 'task_missing')
            return
        c.deadline = c.spec.time_limit
        required = required_capabilities(c.spec)
        if required is None:
            _done(c, effects, 'unsupported_task')
        elif not required <= c.capabilities:
            _done(c, effects, 'unavailable:' + ','.join(sorted(required - c.capabilities)))
        elif c.sim >= c.deadline - c.config.reserve:
            _done(c, effects, 'budget_exhausted')
        else:
            _enter(c, State.VERIFY_EMPTY, effects)
        return
    if c.sim is not None and c.sim >= c.deadline - c.config.reserve:
        _stop(c, effects, 'deadline')
    elif c.active and (c.sim - c.active.sim_start >= c.config.operation_timeout
                       or wall - c.active.wall_start >= c.config.watchdog):
        _stop(c, effects, 'operation_timeout')


def transition(context: Context, event: Event):
    c = deepcopy(context)
    effects = []
    e = event
    if c.state == State.DONE:
        return c, effects
    if e.kind == 'CONNECTION_LOST':
        c.blocked = c.active is not None
        _done(c, effects, str(e.data or 'connection_lost'))
    elif e.kind == 'TICK':
        _tick(c, e, effects)
    elif e.kind == 'SENSOR' and c.hand == Hand.HOLDING and c.state != State.CANCELLING:
        ev = evidence(c, e.data, None)
        if ev is None or ev.hand != Hand.HOLDING:
            c.hand = ev.hand if ev else Hand.UNKNOWN
            _stop(c, effects, 'holding_lost_or_unknown')
    elif e.kind in ('CANCEL', 'ERROR'):
        _stop(c, effects, str(e.data or e.kind.lower()))
    elif e.kind == 'START' and c.state == State.WAIT and c.spec is None:
        c.spec, c.capabilities = e.data
    elif c.active and e.op_id == c.active.id:
        op = c.active
        if e.kind == 'ACCEPTED':
            op.accepted = True
        elif c.state == State.CANCELLING:
            if e.kind == 'CANCEL_ACK':
                if not e.ok:
                    c.blocked = True
                    _done(c, effects, c.reason + ':cancel_refused')
                else:
                    op.cancel_ack = True
            elif e.kind in ('RESULT', 'REJECTED'):
                op.terminal = True
                op.stopped = e.stopped or e.kind == 'REJECTED'
            if c.state != State.DONE and op.cancel_ack and op.terminal:
                c.blocked = not op.stopped
                if op.stopped:
                    c.active = None
                _done(c, effects, c.reason + (':cancelled' if op.stopped else ':stop_unconfirmed'))
        elif e.kind == 'SEND_ERROR':
            _stop(c, effects, 'send_error')
        elif e.kind in ('RESULT', 'REJECTED'):
            if (e.kind == 'RESULT' and not e.ok and not e.stopped
                    and c.state not in VERIFY | {State.OBSERVE, State.DESTINATION}):
                _stop(c, effects, 'execution_failed_stop_unconfirmed')
            else:
                c.active = None
                _result(c, e, op, effects)
    return c, effects


def _result(c, e, op, effects):
    state = c.state
    if e.kind == 'REJECTED' or not e.ok:
        if state in VERIFY:
            c.hand = Hand.UNKNOWN
            _done(c, effects, 'verification_unavailable')
        elif state == State.RETREAT:
            _done(c, effects, 'retreat_failed')
        else:
            _recover(c, effects, str(e.data or 'operation_failed'))
        return
    if state in VERIFY:
        ev = evidence(c, e.data, op.sim_start)
        if ev is None or ev.hand == Hand.UNKNOWN:
            c.hand = Hand.UNKNOWN
            _done(c, effects, 'physical_unknown')
            return
        c.hand = ev.hand
        if state == State.VERIFY_EMPTY:
            if ev.hand != Hand.EMPTY:
                _done(c, effects, 'hand_not_empty')
            else:
                c.target = None
                _enter(c, State.OBSERVE, effects)
        elif state == State.VERIFY_PLACE:
            if not placement(c, ev):
                _done(c, effects, 'placement_unconfirmed')
            else:
                c.history[c.target.id].placed = True
                if c.slot is not None:
                    c.used_slots.add((c.destination, c.slot))
                _enter(c, State.RETREAT, effects)
        elif state == State.RECOVER:
            h = c.history[c.target.id]
            if ev.hand == Hand.EMPTY:
                if h.contacted and not c.target.name:
                    _done(c, effects, 'identity_lost_after_contact')
                else:
                    _enter(c, State.OBSERVE, effects)
            elif h.lift == Lift.LIFTED:
                _enter(c, State.REORIENT if c.spec.variant == 'lying' and not c.reoriented
                       else State.DESTINATION, effects)
            else:
                _done(c, effects, 'lift_unconfirmed')
        elif ev.hand == Hand.EMPTY:
            _recover(c, effects, 'object_lost' if state != State.VERIFY_GRASP else 'empty_grasp')
        elif state == State.VERIFY_GRASP:
            _enter(c, State.LIFT, effects)
        elif state == State.VERIFY_LIFT:
            if not ev.lifted:
                _done(c, effects, 'lift_unconfirmed')
            else:
                c.history[c.target.id].lift = Lift.LIFTED
                _enter(c, State.REORIENT if c.spec.variant == 'lying' else State.DESTINATION, effects)
        elif state == State.VERIFY_REORIENT:
            if not ev.upright:
                _done(c, effects, 'orientation_unconfirmed')
            else:
                c.reoriented = True
                _enter(c, State.DESTINATION, effects)
        elif state == State.VERIFY_TRANSFER:
            if c.scene is None or not fresh(c.scene.stamp, c.sim, c.config.freshness):
                _recover(c, effects, 'stale_destination')
            else:
                _enter(c, State.TRANSFER, effects)
        elif state == State.VERIFY_HOLD:
            _enter(c, State.INSERT if c.spec.kind == 'tubes' else State.LOWER, effects)
        return
    if state in (State.OBSERVE, State.DESTINATION):
        scene = e.data
        if (not isinstance(scene, Scene) or scene.frame != 'pelvis'
                or not fresh(scene.stamp, c.sim, c.config.freshness, op.sim_start)):
            _recover(c, effects, 'stale_scene')
            return
        c.scene = scene
        if state == State.DESTINATION:
            dest = destination_in_scene(c, scene, c.target)
            if dest is None:
                _recover(c, effects, 'destination_missing')
            else:
                c.destination, c.slot = dest
                _enter(c, State.VERIFY_TRANSFER, effects)
        elif c.deadline - c.sim <= c.config.cycle_estimate + c.config.reserve:
            _done(c, effects, 'insufficient_time')
        else:
            obj = select(c, scene)
            if obj is None:
                _done(c, effects, 'candidates_exhausted')
            else:
                c.target = obj
                c.reoriented = False
                c.history.setdefault(obj.id, History()).attempts += 1
                c.destination, c.slot = destination_in_scene(c, scene, obj) or ('', None)
                _enter(c, State.OPEN, effects)
    elif state == State.APPROACH:
        if c.scene is None or not fresh(c.scene.stamp, c.sim, c.config.freshness):
            _recover(c, effects, 'stale_before_contact')
        elif c.deadline - c.sim <= c.config.cycle_estimate + c.config.reserve:
            _done(c, effects, 'insufficient_time')
        else:
            _enter(c, State.DESCEND, effects)
    elif state in NEXT:
        _enter(c, NEXT[state], effects)
