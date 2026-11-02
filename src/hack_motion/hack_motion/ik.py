"""Inverse kinematics of the arms. Owner: motion. Status: NOT IMPLEMENTED (milestone M1).

Plan (see docs/ARCHITECTURE.md, "motion"):
- Model from the robot URDF. The solution container has no simulator files, so either copy
  tiangong_pro_with_hands.urdf into this package (license Apache-2.0, keep attribution) or read /robot_description
  if the simulator publishes it. Decide and note it in README.
- Shoulders hang off waist_yaw_link, so solve in waist_yaw_link and rotate the target by body_yaw_joint:
  one model works both at the table and at the fume hood.
- Damped least squares on the 7 arm joints, joint limits from robot.JOINTS, a null-space pull towards a
  comfortable posture so consecutive solutions do not jump between elbow configurations.
- Candidates: pinocchio (pip "pin"), ikpy, or a hand-written Jacobian solver.
"""


def solve(arm, target_pose, q_seed, waist_yaw):
    """Return 7 joint angles (list, ARM_JOINTS[arm] order) putting the TCP at target_pose (pelvis frame), or None
    if unreachable. target_pose: geometry_msgs/Pose. q_seed: current 7 angles."""
    raise NotImplementedError("IK is not implemented yet (owner: motion, milestone M1)")
