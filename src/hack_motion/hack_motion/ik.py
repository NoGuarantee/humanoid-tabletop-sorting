"""Hand-written IK, the `planner: ik` fallback. Owner: motion. Status: NOT IMPLEMENTED.

cuRobo passed the first-day checks, so MoveTcp goes through curobo_planner.py. This stays only so a parameter
switch back does not require a new module. Do not implement it unless cuRobo stops fitting the GPU budget.
"""


def solve(arm, target_pose, q_seed, waist_yaw):
    """Return 7 joint angles (list, ARM_JOINTS[arm] order) putting the TCP at target_pose (pelvis frame), or None
    if unreachable. target_pose: geometry_msgs/Pose. q_seed: current 7 angles."""
    raise NotImplementedError("IK is not implemented yet (owner: motion, milestone M1)")
