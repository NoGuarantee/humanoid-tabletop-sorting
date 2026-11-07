"""Worker scheduling and fallback regressions without ROS or model weights."""

import os
from queue import Queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hack_perception.classify import merge_trays
from hack_perception.rgbd_labels import CLASS_BY_ID, POOL_IDENTITIES, TRAY_CLASS, validate_detector_labels
from hack_perception.rgbd_worker import _serve, is_fresh


class WorkerTest(unittest.TestCase):
    def test_pose_is_reused_from_one_prepared_scene_even_after_next_prepare_failed(self):
        requests, results = Queue(), Queue()
        original_key = 1, 100
        requests.put(("scene", original_key, ((None, "torso"), None)))
        requests.put(("scene", (1, 110), ("invalid", None)))
        requests.put(("pose", original_key, 4))
        requests.put(None)

        def prepare(_backend, sensors, _mapping):
            if sensors == "invalid":
                raise ValueError("bad depth")
            return "masks", "plane", "trays"

        def pose(_backend, sensors, stamp, pool, *, prepared):
            return sensors, stamp, pool, prepared

        modules = {
            "torch": SimpleNamespace(set_num_threads=lambda threads: None),
            "hack_perception.rgbd.pose_backend": SimpleNamespace(PoseBackend=lambda *a, **k: object()),
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ), \
                patch("hack_perception.rgbd_response.prepare_scene", side_effect=prepare) as prepare_scene, \
                patch("hack_perception.rgbd_response.infer_response", side_effect=pose):
            _serve(requests, results, "/bundle", True, 2)
        self.assertEqual(results.get_nowait()[0], "ready")
        self.assertEqual(results.get_nowait()[:3], ("scene", original_key, ("plane", "trays")))
        self.assertEqual(results.get_nowait()[0], "job_failed")
        self.assertEqual(results.get_nowait()[:3],
                         ("pose", original_key, ((None, "torso"), 100, 4, ("masks", "plane", "trays"))))
        self.assertEqual(prepare_scene.call_count, 2)

    def test_exposure_expiry_checks_source_stamp_and_epoch(self):
        key = 2, 1_000_000_000
        self.assertTrue(is_fresh(key, 2, 2_000_000_000, 1_000_000_000))
        self.assertFalse(is_fresh(key, 2, 2_000_000_001, 1_000_000_000))
        self.assertFalse(is_fresh(key, 3, 1_000_000_000, 1_000_000_000))
        self.assertFalse(is_fresh(key, 2, 999_999_999, 1_000_000_000))

    def test_baseline_trays_fill_missing_ids_even_if_a_model_plane_exists(self):
        mapping = {"green": "center", "blue": "right"}
        good = dict(id="center", color="green", center=(0, 0, 0), yaw=0., size=(1, 1, 1))
        other = dict(good, id="right", color="blue")
        model = SimpleNamespace(**dict(good, center=(1, 2, 3)))
        with patch("hack_perception.classify.find_trays", return_value=[good, other]) as fallback:
            self.assertEqual([t[0] for t in merge_trays([], [], None, mapping)], ["center", "right"])
            merged = merge_trays([model], [], None, mapping)
            self.assertEqual([t[0] for t in merged], ["center", "right"])
            self.assertEqual(merged[0][2], model.center)
            self.assertEqual(fallback.call_count, 2)

    def test_frozen_class_order_matches_trays_and_both_pools(self):
        names = dict(enumerate(("figure", "dish", "box", "electronics", "falcon50", "falcon15", "tray", "rack")))
        validate_detector_labels(names)
        self.assertEqual(CLASS_BY_ID[TRAY_CLASS], "tray")
        self.assertEqual({CLASS_BY_ID[i] for i in POOL_IDENTITIES[4]}, {"figure", "dish", "box", "falcon50", "falcon15"})
        self.assertEqual({CLASS_BY_ID[i] for i in POOL_IDENTITIES[6]}, {"box", "electronics"})
        with self.assertRaisesRegex(ValueError, "class order"):
            validate_detector_labels(names | {1: "box", 2: "dish"})
