"""Pick and place built from motion primitives. Owner: grasping.

Scoring (organisers' answer): only the final position counts. An object never lifted is 0 even if it was pushed or
tipped over on the table, so a failed grasp before the lift costs nothing. Once lifted, the only good outcome is the
right tray (+10); lifted and left on the table, still in the hand or in a wrong tray is -10, on the floor -10 always.
So a skill must know whether it holds the object, and putting it back on the table after a lift does not help.
"""
from .grasp_planner import FINGERTIP, flat_hold_point, plan_grasps, plan_place, pose, tcp_from_local

# The joint loop accepts 0.01 rad. On a 0.7 m arm that is already about 1 cm at the TCP, and the default
# MoveTcp limit is 1 cm, so a descent that had arrived was rejected at 1.02 cm and the hand never closed.
# 2 cm was too far: a later descent timed out 1.86 cm short, still against the object, and the fingers
# closed on nothing. 1.5 cm keeps the arrived attempt and rejects that one.
GRASP_POS_TOL = 0.015


class Skills:
    def __init__(self, clients, db, log):
        self.c = clients
        self.db = db
        self.log = log

    def holding(self, arm, hand_result):
        # TODO(grasping): also check the weight on /arm_6dof_{arm} after lifting (force along gravity)
        return hand_result is not None and hand_result.object_detected

    def pick(self, obj, scene, max_attempts=2):
        """-> the GraspCandidate that holds the object, or None (hand open, object hopefully untouched)."""
        for cand in plan_grasps(obj, scene, self.db)[:max_attempts]:
            arm = cand.arm
            self.c.hand(arm, "open")
            if not self.c.move_tcp(arm, cand.pre_grasp, "direct"):
                continue  # unreachable from this side: nothing touched yet
            if cand.approach_preset != "open" and self.c.hand(arm, cand.approach_preset, cand.force) is None:
                continue
            # A joint-space descent onto a flat pinch swings the open fingers through the dish.
            linear_only = cand.hand_preset == "pinch_flat"
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
            if not self.c.move_tcp(arm, cand.lift, "linear"):
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
        plan = plan_place(cand.arm, container, self.db)
        # Keep the grasp orientation. Turning the wrist to yaw 0 on the way tips a pinched object out.
        q = cand.grasp.orientation
        if cand.hand_preset == "pinch_flat":
            # plan_place aimed a vertical hand. The object sits between the pads, not on the TCP axis.
            local = flat_hold_point(cand.arm)
            for name in ("approach", "release", "retreat"):
                p = getattr(plan, name)
                target = (p.position.x, p.position.y, p.position.z - FINGERTIP)
                setattr(plan, name, tcp_from_local(target, local, q))
        else:
            plan.approach.orientation = q
            plan.release.orientation = q
            plan.retreat.orientation = q
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
        return True
