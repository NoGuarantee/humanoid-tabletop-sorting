"""Deterministic mission reducer. Effects are executed outside this module."""
from copy import deepcopy
import math

from .context import Context, Effect, Event, Hand, History, Lift, Operation, Scene, State
from .policies import destination_in_scene, required_capabilities, select
from .verification import evidence, fresh, placement

VERIFY = {State.VERIFY_EMPTY, State.VERIFY_GRASP, State.VERIFY_LIFT, State.VERIFY_REORIENT,
          State.VERIFY_HOLD, State.VERIFY_TRANSFER, State.VERIFY_PLACE, State.RECOVER}
NEXT = {State.OPEN: State.APPROACH, State.APPROACH: State.DESCEND, State.DESCEND: State.CLOSE,
        State.CLOSE: State.VERIFY_GRASP, State.LIFT: State.VERIFY_LIFT,
        State.REORIENT: State.VERIFY_REORIENT, State.TRANSFER: State.VERIFY_HOLD,
        State.LOWER: State.RELEASE, State.INSERT: State.RELEASE,
        State.RELEASE: State.RETREAT}



def _enter(c, state, effects):
    assert c.active is None and not c.blocked and c.state != State.DONE
    c.state = state
    if state == State.LIFT:
        c.history[c.target.id].lift_attempts += 1
    c.serial += 1
    c.active = Operation(c.serial, state, c.sim, c.wall)
    arm = c.look_side if state == State.LOOK_TABLE else c.start_steps[0][1] if c.start_steps else ''
    effects.append(Effect('RUN', c.serial, state, c.target, c.destination, c.slot, arm))


def _may_have_moved(c, op):
    state = op.phase
    if state in (State.APPROACH, State.TRANSFER):
        c.arm_out = True
    if state == State.DESCEND:
        c.arm_low = True
    if state in (State.DESCEND, State.RELEASE):
        c.hand = Hand.UNKNOWN
        c.scene = None
        c.scene_invalidated_at = op.sim_start
        c.history[c.target.id].contacted = True
    if state == State.LIFT and c.history[c.target.id].lift != Lift.LIFTED:
        c.history[c.target.id].lift = Lift.UNKNOWN


def _look(c, effects):
    """Look at the table, after taking the last arm out of the camera view."""
    _enter(c, State.CLEAR if c.arm_out else State.LOOK_TABLE, effects)


def _side_to_look(c, scene):
    """The side of a tray the task names and no look has shown, not looked at yet; '' when there is none.

    The head camera straight ahead cut off the side trays: perception saw only tray_center, and the objects for the
    others were never picked (eval 2026-11-09). tray_left is on the left (TASKS.md). A side whose arm is disabled is
    not worth the look: nothing goes to its tray."""
    if c.spec is None or c.spec.kind != 'sort':
        return ''
    seen = {x.id for x in scene.containers if x.kind == 'tray'}
    for tray in sorted(set(c.spec.categories.values()) - seen):
        side = 'left' if 'left' in tray else 'right' if 'right' in tray else ''
        if side and side not in c.looked_sides and side != c.disabled_arm:
            return side
    return ''


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


def _free_hands(c, effects, raise_arms=False):
    """Open both hands, then VERIFY_EMPTY. raise_arms: look at the table and raise both arms to ready in between.

    Hands first: motion keeps an arm whose hand is shut where it is. The look comes before the arms: motion refuses
    to plan without the table plane from a perception scene (eval 2026-11-09, ARMS_READY right after the start).
    """
    c.start_steps = [(State.OPEN_HANDS, 'left'), (State.OPEN_HANDS, 'right')]
    if raise_arms:
        c.start_steps += [(State.LOOK_TABLE, ''), (State.OBSERVE, ''),
                          (State.ARMS_READY, 'left'), (State.ARMS_READY, 'right')]
    c.start_steps = [(s, arm) for s, arm in c.start_steps if not arm or arm != c.disabled_arm]
    _enter(c, c.start_steps[0][0], effects)


def _let_go(c, effects):
    """The lift never confirmed: put the object down where it is and go on with the others.

    Ending the mission instead kept the hand shut on the sink, and the next two layouts could not start (eval
    --layouts 3, 2026-11-09). An object that never left the table costs nothing (SCORING, 2 November); one that
    did costs -10 out of a tray, in the hand or not, so letting go never loses points.
    """
    c.history[c.target.id].attempts = c.config.max_attempts
    c.last_recovery = 'lift_unconfirmed'
    _free_hands(c, effects)


def _recover(c, effects, reason):
    c.scene = None
    c.last_recovery = reason
    c.recoveries += 1
    if c.recoveries > c.config.max_recoveries:
        _done(c, effects, 'retries_exhausted:' + reason)
    elif c.target is None:
        _look(c, effects)
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
        if c.start_sim is None:
            c.start_sim = sim
    if wall - c.last_progress_wall >= c.config.watchdog:
        _stop(c, effects, 'clock_missing' if c.sim is None else 'clock_stalled')
        return
    if c.state == State.WAIT:
        if c.spec is None or c.sim is None:
            if wall - c.started_wall >= c.config.watchdog:
                _stop(c, effects, 'task_missing')
            return
        c.deadline = c.start_sim + c.spec.time_limit
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
    elif c.active and c.sim - c.active.sim_start >= c.config.operation_timeout:
        # Sim seconds only: a stalled clock is caught above, and the sim runs slower than the wall clock.
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
            _may_have_moved(c, op)
        elif c.state == State.CANCELLING:
            if e.kind == 'CANCEL_ACK':
                if not e.ok:
                    c.blocked = True
                    _done(c, effects, c.reason + ':cancel_refused')
                else:
                    op.cancel_ack = True
            elif e.kind in ('RESULT', 'REJECTED'):
                op.terminal = True
                # A sensor-only op moves nothing: its reading, already queued when the hold was lost, is a stop.
                # Taken as unconfirmed, it ended a RECOVER with 300 s and 8 objects left (eval 2026-11-08).
                op.stopped = (e.stopped or e.kind == 'REJECTED'
                              or op.phase in VERIFY | {State.OBSERVE, State.DESTINATION})
            if c.state != State.DONE and op.cancel_ack and op.terminal:
                c.blocked = not op.stopped
                if op.stopped:
                    c.active = None
                if op.stopped and c.reason == 'holding_lost_or_unknown' and c.target is not None:
                    # A hold lost mid-move, with the arm measured at rest, leaves the rest of the table to sort:
                    # the sink slipping on the lift ended a mission with 400 s and 8 objects left (eval 2026-11-08).
                    # RECOVER measures the hand again before anything moves.
                    c.reason = ''
                    _recover(c, effects, 'object_lost')
                else:
                    _done(c, effects, c.reason + (':cancelled' if op.stopped else ':stop_unconfirmed'))
        elif e.kind == 'SEND_ERROR':
            _may_have_moved(c, op)
            _stop(c, effects, 'send_error')
        elif e.kind in ('RESULT', 'REJECTED'):
            if e.kind == 'RESULT':
                _may_have_moved(c, op)
            if (e.kind == 'RESULT' and not e.ok and not e.stopped
                    and c.state not in VERIFY | {State.OBSERVE, State.DESTINATION, State.OPEN_HANDS}):
                _stop(c, effects, 'execution_failed_stop_unconfirmed')
            else:
                c.active = None
                _result(c, e, op, effects)
    return c, effects


def _result(c, e, op, effects):
    state = c.state
    if c.start_steps:
        # Steps that free the hands (see _free_hands). Done or not, VERIFY_EMPTY measures the hands after them and
        # ends the mission if one still holds something.
        c.start_steps = c.start_steps[1:]
        _enter(c, c.start_steps[0][0] if c.start_steps else State.VERIFY_EMPTY, effects)
        return
    if state == State.BACK_OFF:
        # Up or not, look again: a plan from a low arm fails on its own and spends the recovery budget.
        if e.kind == 'REJECTED' or not e.ok:
            c.last_recovery = 'back_off_failed'
        _look(c, effects)
        return
    if state == State.CLEAR:
        # Once per look: an arm that cannot reach the side pose may still leave the view clear enough.
        c.arm_out = False
        if e.kind == 'REJECTED' or not e.ok:
            c.last_recovery = 'clear_failed'
        _enter(c, State.LOOK_TABLE, effects)
        return
    if state == State.LIFT and e.kind == 'RESULT' and e.ok:
        c.arm_low = False
    if e.kind == 'REJECTED' or not e.ok:
        if state in VERIFY:
            c.hand = Hand.UNKNOWN
            _done(c, effects, 'verification_unavailable')
        elif state == State.RETREAT:
            # The hand is open and placed is never set without the camera: CLEAR plans its own way out of the
            # tray, and the look decides. Ending here lost 8 objects after a sorted cartridge (eval 2026-11-08).
            c.last_recovery = 'retreat_failed: ' + str(e.data or '')
            c.place_look = True
            _look(c, effects)
        elif state == State.LOWER:
            # TRANSFER left the fingertips 0.2 m above the tray opening: open there. A straight descent that did
            # not plan failed the same way six times, and the mission ended with the cup in the hand, -10 (seed
            # 146, 2026-11-10). Let go over the tray, it lands inside or costs what keeping it would.
            c.last_recovery = 'lower_failed: ' + str(e.data or '')
            _enter(c, State.RELEASE, effects)
        else:
            _recover(c, effects, str(e.data or 'operation_failed'))
        return
    if state == State.LOOK_TABLE:
        c.scene_invalidated_at = c.sim
        _enter(c, State.VERIFY_PLACE if c.place_look else State.OBSERVE, effects)
        c.place_look = False
        return
    if state == State.RETREAT:
        # The arm above the tray hid it and the object from the head camera, so the placement went unconfirmed
        # (eval 2026-11-08). Take the arm out of view first, then look at the placement.
        c.place_look = True
        _look(c, effects)
        return
    if state in VERIFY:
        ev = evidence(c, e.data, op.sim_start)
        if ev is None or ev.hand == Hand.UNKNOWN:
            c.hand = Hand.UNKNOWN
            if state == State.VERIFY_EMPTY and not c.start_cleared:
                # The previous layout leaves the robot as it was. A hand shut on a sink read UNKNOWN and two
                # layouts ended 6 s after the start (eval --layouts 3, 2026-11-09). An opened hand still pressed on
                # the new layout (fingers 1.32 1.39 past open, the rest creeping) read UNKNOWN again: raise the arms
                # too, then measure again.
                c.start_cleared = True
                _free_hands(c, effects, raise_arms=True)
            elif state == State.RECOVER and c.recoveries < c.config.max_recoveries:
                # A hand still settling after a failed move: measure again, within the recovery budget.
                c.recoveries += 1
                c.last_recovery = 'hand_unknown'
                _enter(c, State.RECOVER, effects)
            elif state == State.VERIFY_GRASP and c.recoveries < c.config.max_recoveries:
                # Nothing is lifted yet; RECOVER measures the hand again and backs off. Ending here cost the
                # rest of the mission twice over a finger pushed past its range (eval 2026-11-08).
                _recover(c, effects, 'hand_unknown')
            elif c.unknown_frees < c.config.max_unknown_frees:
                # Open both hands and raise both arms, then go on with the others. Ending here lost 222 s after
                # an open hand pressed on a box it never closed on (eval --layouts 3, 2026-11-09). A lifted object
                # that falls costs -10, as it does left in the hand at the end; over the tray it lands inside.
                c.unknown_frees += 1
                c.last_recovery = 'hand_unknown_freed'
                if c.target is not None and state == State.VERIFY_PLACE:
                    c.history[c.target.id].released = True  # let go over the tray: it may be in it
                c.arm_low = False
                _free_hands(c, effects, raise_arms=True)
            elif state == State.VERIFY_EMPTY and ev is not None and len(ev.empty_arms) == 1 and not c.disabled_arm:
                # Every free left one hand shut (little finger and thumb at effort 1.0, a layout ended with 374 s
                # left, eval 2026-11-09). Go on with the other hand; select skips what only the shut one would take.
                c.disabled_arm = 'right' if 'left' in ev.empty_arms else 'left'
                c.last_recovery = 'arm_disabled:' + c.disabled_arm
                c.hand = Hand.EMPTY
                c.target = None
                _look(c, effects)
            else:
                _done(c, effects, 'physical_unknown')
            return
        c.hand = ev.hand
        if state == State.VERIFY_EMPTY:
            if ev.hand != Hand.EMPTY:
                _done(c, effects, 'hand_not_empty')
            else:
                c.target = None
                _look(c, effects)
        elif state == State.VERIFY_PLACE:
            if not placement(c, ev) and ev.hand == Hand.EMPTY:
                # Released over the destination but not seen there: the camera missed the tray behind the arm
                # once, and a cartridge sorted correctly ended the mission with 8 objects left (eval 2026-11-08).
                # The hand is empty, so go on with the others, and never pick this one again: it may be in a tray.
                c.history[c.target.id].released = True
                _enter(c, State.VERIFY_EMPTY, effects)
            elif not placement(c, ev):
                _done(c, effects, 'placement_unconfirmed')
            else:
                c.history[c.target.id].placed = True
                if c.slot is not None:
                    c.used_slots.add((c.destination, c.slot))
                _enter(c, State.VERIFY_EMPTY, effects)
        elif state == State.RECOVER:
            h = c.history[c.target.id]
            if ev.hand == Hand.HOLDING and ev.lifted:
                h.lift = Lift.LIFTED
            if ev.hand == Hand.EMPTY:
                if h.contacted and not c.target.name:
                    _done(c, effects, 'identity_lost_after_contact')
                elif c.arm_low:
                    c.arm_low = False
                    _enter(c, State.BACK_OFF, effects)
                else:
                    _look(c, effects)
            elif h.lift == Lift.LIFTED:
                _enter(c, State.REORIENT if c.spec.variant == 'lying' and not c.reoriented
                       else State.DESTINATION, effects)
            elif h.lift_attempts < c.config.max_attempts:
                _enter(c, State.LIFT, effects)
            else:
                _let_go(c, effects)
        elif ev.hand == Hand.EMPTY:
            _recover(c, effects, 'object_lost' if state != State.VERIFY_GRASP else 'empty_grasp')
        elif state == State.VERIFY_GRASP:
            _enter(c, State.LIFT, effects)
        elif state == State.VERIFY_LIFT:
            if not ev.lifted:
                if c.history[c.target.id].lift_attempts < c.config.max_attempts:
                    _enter(c, State.LIFT, effects)
                else:
                    _let_go(c, effects)
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
                or not fresh(scene.stamp, c.sim, c.config.freshness, c.scene_invalidated_at)):
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
        elif c.look_side or _side_to_look(c, scene):
            # A side look only adds trays: the next look is the other side or straight ahead, where the objects are.
            c.look_side = _side_to_look(c, scene)
            c.looked_sides.add(c.look_side)
            c.scene = None
            _look(c, effects)
        else:
            obj = select(c, scene)
            if obj is None and c.empty_observations < c.config.max_empty_observations:
                c.empty_observations += 1
                c.scene = None
                _look(c, effects)
            elif obj is None:
                _done(c, effects, 'candidates_exhausted')
            else:
                c.empty_observations = 0
                # Recoveries are budgeted per attempt: five 'no /joint_states yet' rejections while motion started
                # spent a mission-wide budget, and one empty grasp later ended the mission (eval 2026-11-08).
                # Attempts per object and the sim deadline still bound the mission.
                c.recoveries = 0
                c.target = obj
                c.reoriented = False
                c.history.setdefault(obj.id, History()).attempts += 1
                c.destination, c.slot = destination_in_scene(c, scene, obj) or ('', None)
                _enter(c, State.OPEN, effects)
    elif state in NEXT:
        _enter(c, NEXT[state], effects)
