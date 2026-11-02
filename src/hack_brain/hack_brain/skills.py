"""Pick and place built from motion primitives. Owner: grasping.

Scoring (organisers' answer): only the final position counts. An object never lifted is 0 even if it was pushed or
tipped over on the table, so a failed grasp before the lift costs nothing. Once lifted, the only good outcome is the
right tray (+10); lifted and left on the table, still in the hand or in a wrong tray is -10, on the floor -10 always.
So a skill must know whether it holds the object, and putting it back on the table after a lift does not help.
"""
from .grasp_planner import plan_grasps, plan_place


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
            if not self.c.move_tcp(arm, cand.grasp, "linear"):
                self.c.move_tcp(arm, cand.pre_grasp, "linear")
                continue
            hr = self.c.hand(arm, cand.hand_preset, cand.force)
            if not self.holding(arm, hr):
                self.c.hand(arm, "open")
                self.c.move_tcp(arm, cand.pre_grasp, "linear")
                continue
            self.c.move_tcp(arm, cand.lift, "linear")
            # TODO(grasping): re-check holding after the lift (slipped objects)
            return cand
        return None

    def place(self, cand, container, put_back_pose=None):
        plan = plan_place(cand.arm, container, self.db)
        if not self.c.move_tcp(cand.arm, plan.approach, "direct"):
            # TODO(grasping): cannot reach the tray -> another approach or the other arm. Putting the object back
            # gives the same -10 as dropping it, but frees the arm for a retry later.
            self.log.error("tray unreachable while holding an object")
            return False
        self.c.move_tcp(cand.arm, plan.release, "linear")
        self.c.hand(cand.arm, "open")
        self.c.move_tcp(cand.arm, plan.retreat, "linear")
        return True
