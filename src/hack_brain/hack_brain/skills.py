"""Pick and place built from motion primitives. Owner: grasping.

Two ways in. The mission FSM calls `command`: one action goal per phase, the FSM owns the sequence. `pick`, `place`
and `ready` are blocking sequences with the fallbacks found on the bench (joint-space descent after a failed straight
one, lift in two steps by the table edge, carry up and across). tools/pick_one.py runs them through
blocking.BlockingClients; the FSM does not have them yet.

Scoring (organisers' answer): only the final position counts. An object never lifted is 0 even if it was pushed or
tipped over on the table, so a failed grasp before the lift costs nothing. Once lifted, the only good outcome is the
right tray (+10); lifted and left on the table, still in the hand or in a wrong tray is -10, on the floor -10 always.
So a skill must know whether it holds the object, and putting it back on the table after a lift does not help.

Tube insertion/reorientation deliberately have no fabricated geometric implementation.
"""
from geometry_msgs.msg import Pose

from .context import State
from .grasp_planner import FINGERTIP, flat_hold_point, plan_grasps, plan_place, pose, tcp_from_local
from . import recorded_grasps

# The joint loop accepts 0.01 rad. On a 0.7 m arm that is already about 1 cm at the TCP, and the default
# MoveTcp limit is 1 cm, so a descent that had arrived was rejected at 1.02 cm and the hand never closed.
# 2 cm was too far: a later descent timed out 1.86 cm short, still against the object, and the fingers
# closed on nothing. 1.5 cm keeps the arrived attempt and rejects that one.
GRASP_POS_TOL = 0.015
# Head yaw for a side look. The side trays sit near the 43 degree edge of the 86 degree head camera straight ahead;
# turned 0.6 rad (34 degrees) the frame covers from -9 to 77 degrees.
HEAD_SIDE_YAW = 0.6
# Pre-grasp and above-tray poses are in free air, and the linear move after them plans from the measured pose.
# An approach that stopped 1.01 cm off was rejected at the 1 cm default and cost the attempt (eval 2026-11-08).
FREE_POS_TOL = 0.02


def place_for(cand, container, db):
    """plan_place for the hand as it holds the object.

    Keep the grasp orientation: turning the wrist to yaw 0 on the way tips a pinched object out. plan_place aims
    a vertical hand; a leaned (flat pinch) hand holds the object between the pads, not on the TCP axis.
    """
    if isinstance(cand, recorded_grasps.RecordedGrasp):
        return recorded_grasps.place(cand, container, db)
    plan = plan_place(cand.arm, container, db)
    q = cand.grasp.orientation
    if cand.leaned:
        local = flat_hold_point(cand.arm)
        for name in ("approach", "release", "retreat"):
            p = getattr(plan, name)
            target = (p.position.x, p.position.y, p.position.z - FINGERTIP)
            setattr(plan, name, tcp_from_local(target, local, q))
    else:
        plan.approach.orientation = q
        plan.release.orientation = q
        plan.retreat.orientation = q
    return plan


class Skills:
    def __init__(self, db, clients=None, log=None):
        """`clients` and `log` only for the blocking pick/place/ready (blocking.BlockingClients)."""
        self.db = db
        self.c = clients
        self.log = log
        self.grasp = None
        self.place_plan = None

    def needs_model_pose(self, effect):
        return (effect.phase in (State.OPEN, State.DESCEND) and effect.target is not None
                and recorded_grasps.uses_recorded(self.db, effect.target.name))

    def prepare_recorded(self, effect, scene, data, now, after_stamp_ns, attempt=1, disabled_arm=''):
        obj = recorded_grasps.selected_object(scene, effect.target)
        container = next((x for x in scene.containers if x.id == effect.destination and x.kind == 'tray'), None)
        key = self.grasp.profile_key if effect.phase == State.DESCEND else None
        cand = recorded_grasps.candidate(data, now, obj, container, self.db, after_stamp_ns, attempt, key, disabled_arm)
        self.grasp = recorded_grasps.refreshed(self.grasp, cand) if effect.phase == State.DESCEND else cand

    def command(self, effect, scene, attempt=1):
        from hack_interfaces.action import Hand, MoveJoints, MoveTcp
        phase = effect.phase
        if phase == State.LOOK_TABLE:
            return 'joints', MoveJoints.Goal(
                joints=['head_pitch_joint', 'head_yaw_joint', 'body_yaw_joint'],
                positions=[0.4, {'left': HEAD_SIDE_YAW, 'right': -HEAD_SIDE_YAW}.get(effect.arm, 0.0), 0.0])
        if phase == State.OPEN_HANDS:  # at the start: the previous layout may have left a hand shut
            return 'hand', Hand.Goal(hand=effect.arm, preset='open', force=0.0)
        if phase == State.ARMS_READY:  # then up off the table and whatever the new layout put under the hand
            return 'tcp', MoveTcp.Goal(arm=effect.arm, target=Pose(), mode='ready')
        if phase == State.OPEN:
            obj = (recorded_grasps.selected_object(scene, effect.target) if self.needs_model_pose(effect) else
                   next(o for o in scene.objects if (o.name or o.track_id) == effect.target.id))
            container = next((x for x in scene.containers if x.id == effect.destination and x.kind == 'tray'), None)
            self.place_plan = None
            if recorded_grasps.uses_recorded(self.db, obj.name):
                if not isinstance(self.grasp, recorded_grasps.RecordedGrasp):
                    raise ValueError('missing_recorded_model_pose')
            else:
                candidates = plan_grasps(obj, scene, self.db, container)
                if not candidates:
                    raise ValueError('no_grasp_candidates')
                self.grasp = candidates[min(attempt - 1, len(candidates) - 1)]
            # The pinch turns the thumb across before the descent (pinch_ready), as pick() does. Turned at the
            # grasp instead, the thumb pushed the cartridge aside and it slipped out on the lift (sort3 seed 1).
            return 'hand', Hand.Goal(hand=self.grasp.arm, preset=self.grasp.approach_preset,
                                     force=float(self.grasp.force if self.grasp.approach_preset != 'open' else 0.0))
        if self.grasp is None:
            raise ValueError('missing_grasp_plan')
        g = self.grasp
        if phase == State.CLEAR:  # the arm of the last grasp, to the side and out of the head camera view
            return 'tcp', MoveTcp.Goal(arm=g.arm, target=Pose(), mode='ready')
        if phase in (State.CLOSE, State.RELEASE):
            release = (g.profile_key + '_release' if isinstance(g, recorded_grasps.RecordedGrasp)
                       and 'release' in recorded_grasps.PROFILES[g.profile_key] else 'open')
            preset = g.hand_preset if phase == State.CLOSE else release
            return 'hand', Hand.Goal(hand=g.arm, preset=preset, force=float(g.force if preset != 'open' else 0.0))
        if phase == State.TIGHTEN:
            return 'hand', Hand.Goal(hand=g.arm, preset=g.profile_key + '_postlift', force=float(g.force))
        if phase == State.TRANSFER:
            container = next(x for x in scene.containers if x.id == effect.destination and x.kind == 'tray')
            self.place_plan = place_for(g, container, self.db)
        poses = {State.APPROACH: g.pre_grasp, State.DESCEND: g.grasp, State.LIFT: g.lift, State.BACK_OFF: g.pre_grasp}
        if self.place_plan:
            poses.update({State.TRANSFER: self.place_plan.approach, State.LOWER: self.place_plan.release,
                          State.RETREAT: self.place_plan.retreat})
        if phase not in poses:
            raise NotImplementedError('skill unavailable: ' + phase.value)
        recorded = isinstance(g, recorded_grasps.RecordedGrasp)
        mode = {State.APPROACH: 'direct', State.TRANSFER: 'carry'}.get(phase, 'linear')
        contact = {}
        if recorded and phase == State.DESCEND:
            contact = dict(contact_object=g.contact_object_id, contact_center=list(g.observed_center),
                           contact_stamp_ns=g.exposure['stamp_ns'])
            if not g.contact_object_id:
                raise ValueError('missing_recorded_contact_object')
        return 'tcp', MoveTcp.Goal(arm=g.arm, target=poses[phase],
                                  mode=mode + '_fixed' if recorded else mode,
                                  pos_tol=.005 if recorded else {State.DESCEND: GRASP_POS_TOL, State.APPROACH: FREE_POS_TOL,
                                           State.TRANSFER: FREE_POS_TOL}.get(phase, 0.0),
                                  rot_tol=.05 if recorded else 0.0, **contact)

    def holding(self, arm, hand_result):
        # TODO(grasping): also check the weight on /arm_6dof_{arm} after lifting (force along gravity)
        return hand_result is not None and hand_result.object_detected

    def pick(self, obj, scene, max_attempts=2):
        """-> the GraspCandidate that holds the object, or None (hand open, object hopefully untouched)."""
        tried = 0
        for cand in plan_grasps(obj, scene, self.db):
            if tried >= max_attempts:
                break
            arm = cand.arm
            self.c.hand(arm, "open")
            if not self.c.move_tcp(arm, cand.pre_grasp, "direct"):
                continue  # unreachable from this side: nothing touched yet, does not count
            tried += 1
            if cand.approach_preset != "open" and self.c.hand(arm, cand.approach_preset, cand.force) is None:
                continue
            # A joint-space descent onto a flat pinch swings the open fingers through the dish.
            linear_only = cand.leaned
            if not self._descend(arm, cand.grasp, linear_only):
                self.c.hand(arm, "open")
                self.c.move_tcp(arm, cand.pre_grasp, "linear")
                continue
            hr = self.c.hand(arm, cand.hand_preset, cand.force)
            if not self.holding(arm, hr):
                self.log.warn(f"hand {arm} closed without an object ({self._fingers(hr)})")
                self.c.hand(arm, "open")
                self.c.move_tcp(arm, cand.pre_grasp, "linear")
                continue
            self.log.info(f"hand {arm} closed on an object ({self._fingers(hr)})")
            if not self._lift(arm, cand):
                self.log.warn(f"lift failed; opening {arm}")
                self.c.hand(arm, "open")
                if not self.c.move_tcp(arm, cand.pre_grasp, "linear"):
                    self.c.move_tcp(arm, cand.pre_grasp, "direct")
                continue
            # The close can register a brush. After the lift the fingers finish their travel if nothing came up.
            held = self.c.hand(arm, cand.hand_preset, cand.force)
            if not self.holding(arm, held):
                self.log.warn(f"object slipped on the lift ({self._fingers(held)})")
                self.c.hand(arm, "open")
                continue
            return cand
        return None

    @staticmethod
    def _fingers(hr):
        if hr is None or not hr.position:
            return ""
        return " ".join(f"{p:.2f}" for p in hr.position)

    def _lift(self, arm, cand):
        """Straight up with the object. A 0.15 m straight line from a grasp by the table edge (x 0.25) had no plan
        while the 0.12 m one back to the pre-grasp did (2026-11-07): then that first, and the rest from there.
        The pre-grasp is already clear of the table, so the second step is not required."""
        if self.c.move_tcp(arm, cand.lift, "linear"):
            return True
        if cand.pre_grasp.position.z <= cand.grasp.position.z or not self.c.move_tcp(arm, cand.pre_grasp, "linear"):
            return False
        if cand.lift.position.z > cand.pre_grasp.position.z:
            self.c.move_tcp(arm, cand.lift, "linear")
        return True

    def _descend(self, arm, grasp, linear_only=False):
        """Down to the grasp. A straight line often meets the body; the same pose can still be free in joint space."""
        if self.c.move_tcp(arm, grasp, "linear", pos_tol=GRASP_POS_TOL):
            return True
        if linear_only:
            return False
        return self.c.move_tcp(arm, grasp, "direct", pos_tol=GRASP_POS_TOL)

    def _to_approach(self, arm, approach):
        """Reach the pose above the tray. A direct path from the grasp often has none: the wrist is by its stop.

        The high pose in front of the chest is one direct moves already reach with this orientation.
        """
        if self.c.move_tcp(arm, approach, "direct"):
            return True
        y = -0.30 if arm == "right" else 0.30
        high = pose(0.40, y, max(approach.position.z, 0.40), approach.orientation)
        return self.c.move_tcp(arm, high, "direct") and self.c.move_tcp(arm, approach, "direct")

    def _carry(self, arm, lift, approach):
        """Palm-down, in straight lines: up, across, and only then to the approach.

        A joint-space path from the grasp rolls the hand on the way and a pinch lets go. The release itself
        stays a separate step: straight down into the tray meets the solid box.
        """
        q = approach.orientation
        high = approach.position.z
        up = pose(lift.position.x, lift.position.y, high, q)
        over = pose(approach.position.x, approach.position.y, high, q)
        return self.c.move_tcp(arm, up, "linear") and self.c.move_tcp(arm, over, "linear")

    def place(self, cand, container, put_back_pose=None):
        plan = place_for(cand, container, self.db)
        if not self._carry(cand.arm, cand.lift, plan.approach) and not self._to_approach(cand.arm, plan.approach):
            # TODO(grasping): another approach or the other arm. Putting the object back gives the same -10
            # as dropping it, but frees the arm for a retry later.
            self.log.error("tray unreachable while holding an object")
            return False
        # The lower pose drives the forearm through the solid tray, and a joint-space detour starts by jumping
        # the arm off the self-collision buffer, which throws the object. Open where the carry arrived.
        if not self.c.move_tcp(cand.arm, plan.release, "linear"):
            self.log.warn("release pose not reached; opening above the tray")
        self.c.hand(cand.arm, "open")
        self.c.move_tcp(cand.arm, plan.retreat, "linear")
        self.ready(cand.arm)
        return True

    def ready(self, arm):
        """Arm out to the side (robot.ready_pose) by a collision-free joint path, so the next plan starts from
        known joints and the head camera sees the table.

        After the carry the arm can be one IK branch away from any table pose: the next pre-grasp had no plan.
        The target pose is not used in this mode.
        """
        return self.c.move_tcp(arm, Pose(), "ready")
