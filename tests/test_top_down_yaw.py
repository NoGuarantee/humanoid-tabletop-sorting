"""robot.tilt / yaw / top_down: what a carry may turn a top-down hand by without tipping the object."""
import math
import unittest

from hack_motion.robot import VERTICAL, tilt, top_down, yaw


def rpy_quat(roll, pitch, heading):
    """(x, y, z, w), as grasp_planner.quat_from_rpy."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(heading / 2), math.sin(heading / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class TopDownYaw(unittest.TestCase):
    def test_a_top_down_hand_has_no_tilt_at_any_yaw(self):
        for heading in (-2.5, -0.92, 0.0, 1.37, 3.1):
            self.assertAlmostEqual(tilt(top_down(heading)), 0.0, places=6)
            self.assertAlmostEqual(yaw(top_down(heading)), heading, places=6)

    def test_the_cup_goal_of_seed_13051_is_top_down(self):
        q = (0.0, 0.0, -0.442, 0.897)
        self.assertLess(tilt(q), VERTICAL)
        self.assertAlmostEqual(math.degrees(yaw(q)), -52.5, delta=0.5)

    def test_the_leaned_pinch_is_not_top_down(self):
        q = rpy_quat(math.radians(32), 0.0, 0.7)
        self.assertAlmostEqual(tilt(q), math.radians(32), places=6)
        self.assertGreater(tilt(q), VERTICAL)

    def test_tilt_ignores_the_yaw(self):
        for heading in (-1.0, 0.5, 2.0):
            self.assertAlmostEqual(tilt(rpy_quat(0.0, 0.2, heading)), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
