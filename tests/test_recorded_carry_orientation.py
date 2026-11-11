"""Bounded placement turns include permitted error from the preceding leg."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hack_motion.recorded_carry import orientation_arc_deviation


def quat(vector):
    return Rotation.from_rotvec(vector).as_quat()[[3, 0, 1, 2]]


class CarryOrientationTests(unittest.TestCase):
    def test_measured_start_lag_does_not_count_as_uncommanded_rotation(self):
        # The 4090 replay rejected the next 0.04-rad segment at 0.050211 rad
        # from its goal, although the path was making the requested turn.
        start, finish = quat([0, 0, -.010211]), quat([0, 0, .04])
        path = [quat([0, 0, z]) for z in np.linspace(-.010211, .04, 31)]
        self.assertLess(orientation_arc_deviation(path, start, finish, .05), 1e-7)

    def test_unrequested_tilt_and_overshoot_remain_visible(self):
        start, finish = quat([0, 0, 0]), quat([0, 0, .04])
        self.assertGreater(orientation_arc_deviation([quat([.08, 0, 0])], start, finish, .05), .079)
        self.assertAlmostEqual(orientation_arc_deviation([quat([0, 0, .10])], start, finish, .05), .06)
        with self.assertRaisesRegex(ValueError, 'rotation_step_exceeded'):
            orientation_arc_deviation([start], start, quat([0, 0, .2]), .05)

    def test_stationary_orientation_and_quaternion_sign(self):
        q = quat([.4, -.3, 2.1])
        self.assertLess(orientation_arc_deviation([-q, q], q, -q, .05), 1e-7)
        tilted = (Rotation.from_rotvec([.08, 0, 0]) * Rotation.from_quat(q[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
        self.assertAlmostEqual(orientation_arc_deviation([tilted], q, q, .05), .08)

    def test_analytic_arc_distance_matches_dense_rotation_samples(self):
        rng = np.random.default_rng(146)
        for _ in range(20):
            a = Rotation.random(random_state=rng)
            b = Rotation.from_rotvec(rng.normal(size=3)*.015) * a
            path = Rotation.from_rotvec(rng.normal(size=(12, 3))*.06) * a
            arc = Slerp([0, 1], Rotation.from_quat([a.as_quat(), b.as_quat()]))(np.linspace(0, 1, 2001))
            dense = max(np.min((q.inv() * arc).magnitude()) for q in path)
            got = orientation_arc_deviation(path.as_quat()[:, [3, 0, 1, 2]],
                                            a.as_quat()[[3, 0, 1, 2]], b.as_quat()[[3, 0, 1, 2]], .05)
            self.assertAlmostEqual(got, dense, delta=1e-6)
