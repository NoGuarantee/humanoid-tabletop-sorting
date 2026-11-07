"""Contract regressions that do not require ROS, weights or a simulator."""
from dataclasses import replace
import json
from types import SimpleNamespace as Record
import unittest
from unittest.mock import patch

import numpy as np

from hack_perception.rgbd.detection import Detection
from hack_perception.rgbd_frames import CAMERAS, STREAMS, PairBuffer, decode_pair, decode_view, stamp_ns
from hack_perception.rgbd_response import SCHEMA, infer_response, object_response, response


def frames(ns, frame):
    header = Record(frame_id=frame, stamp=Record(sec=ns // 10**9, nanosec=ns % 10**9))
    info = Record(header=header, height=1, width=2, k=[4., 0., 1., 0., 4., .5, 0., 0., 1.], d=[])
    return dict(color=Record(header=header, height=1, width=2, step=7, encoding="rgb8", data=bytes(range(7))),
                depth=Record(header=header, height=1, width=2, step=5, encoding="16UC1",
                             is_bigendian=True, data=b"\x01\x02\x03\x04\xff"),
                color_info=info, depth_info=info)


class ModelPoseInputsTest(unittest.TestCase):
    def test_pair_requires_all_eight_messages_at_one_fresh_stamp(self):
        buffer = PairBuffer(capacity=2)
        for camera in CAMERAS:
            view = frames(100, camera)
            for kind in STREAMS:
                if (camera, kind) != ("cam_torso", "depth_info"):
                    buffer.add(camera, kind, view[kind])
        buffer.add("cam_torso", "depth_info", frames(101, "cam_torso")["depth_info"])
        self.assertIsNone(buffer.latest(now_ns=110, max_age_ns=100))
        head, torso = buffer.at(100)
        self.assertIsNotNone(head)
        self.assertIsNone(torso)
        buffer.add("cam_torso", "depth_info", frames(100, "cam_torso")["depth_info"])
        self.assertEqual(buffer.latest(now_ns=110, max_age_ns=10)[0], 100)
        self.assertIsNone(buffer.latest(now_ns=111, max_age_ns=10))
        self.assertIsNone(buffer.latest(now_ns=99, max_age_ns=10))
        buffer.clear()
        self.assertIsNone(buffer.latest(now_ns=110, max_age_ns=10))

    def test_pair_queries_tf_for_each_camera_at_the_exposure(self):
        queries = []

        def lookup(header):
            queries.append((header.frame_id, stamp_ns(header)))
            return np.eye(4)

        sensors = decode_pair([frames(100, cam) for cam in CAMERAS], lookup)
        self.assertEqual(queries, [("cam_head", 100), ("cam_torso", 100)])
        np.testing.assert_equal(sensors[0]["T"], np.eye(4))
        with self.assertRaisesRegex(ValueError, "distinct optical frames"):
            decode_pair([frames(100, "cam_head")] * 2, lookup)

    def test_decode_handles_padded_big_endian_depth(self):
        sensor = decode_view(frames(100, "camera"))
        np.testing.assert_equal(sensor["depth"], [[258, 772]])
        np.testing.assert_equal(sensor["rgb"], [[[0, 1, 2], [3, 4, 5]]])
        self.assertEqual(sensor["depth"].dtype, np.uint16)

    def test_decode_rejects_mismatched_calibration(self):
        view = frames(100, "camera")
        view["depth_info"] = Record(**vars(view["depth_info"]))
        view["depth_info"].k = [5., *view["color_info"].k[1:]]
        with self.assertRaisesRegex(ValueError, "matching pinhole intrinsics"):
            decode_view(view)

    def test_decode_rejects_bad_payload(self):
        view = frames(100, "camera")
        view["depth"].data = b""
        with self.assertRaisesRegex(ValueError, "payload length"):
            decode_view(view)


class ModelPoseResponseTest(unittest.TestCase):
    def setUp(self):
        self.candidate = Detection("sink", 1, np.eye(4), np.zeros(3), np.ones(3),
                                   .99, .001, 2, "full", True, False, 400, (1, 2))

    def test_accepts_a_valid_pose_from_either_camera(self):
        for slot, camera in enumerate(CAMERAS):
            with self.subTest(camera=camera):
                instances = (1, None) if slot == 0 else (None, 1)
                candidate = replace(self.candidate, views=1, instances=instances)
                backend = Record(estimate=lambda *args: [candidate], pool_categories=lambda pool: {1})
                segmentation = [None, None]
                segmentation[slot] = (None, [
                    {"class_id": 1, "instance_id": 1, "confidence": .99},
                    {"class_id": 1, "instance_id": 2, "confidence": .80},
                ], None)
                sensors = [object() if i == slot else None for i in range(2)]
                with patch("hack_perception.rgbd_response.observed_regions", return_value=[]):
                    payload = infer_response(backend, sensors, 500, 4, prepared=(segmentation, None, []))
                good, no_depth = payload["objects"]
                self.assertEqual(good["pose_status"], "ok")
                self.assertEqual(good["depth_supported_views"], 1)
                self.assertEqual(good["instances"], {camera: 1})
                self.assertEqual(good["T_pelvis_object"], np.eye(4).tolist())
                self.assertEqual(no_depth["pose_status"], "insufficient_depth")
                self.assertEqual(no_depth["instances"], {camera: 2})

    def test_still_rejects_unsupported_clipped_or_uncertain_poses(self):
        for detection, status in ((replace(self.candidate, views=0), "insufficient_depth"),
                                  (replace(self.candidate, views=1, clipped=True), "reacquire_clipped"),
                                  (replace(self.candidate, views=1, valid=False), "reacquire_uncertain_fit"),
                                  (replace(self.candidate, views=2, valid=False), "reacquire_uncertain_fit")):
            with self.subTest(status=status):
                out = object_response(detection)
                self.assertEqual(out["pose_status"], status)
                self.assertIsNone(out["T_pelvis_object"])
                self.assertIsNone(out["center_pelvis_m"])

    def test_payload_preserves_schema_exposure_frame_and_accepted_model_transform(self):
        obj = object_response(self.candidate)
        payload = response("ok", stamp=500, pool=4, objects=[obj])
        self.assertEqual(payload["schema"], SCHEMA)
        self.assertEqual(payload["stamp_ns"], 500)
        self.assertEqual(payload["frame_id"], "pelvis")
        self.assertEqual(payload["objects"][0]["pose_status"], "ok")
        self.assertEqual(payload["objects"][0]["instances"], {"cam_head": 1, "cam_torso": 2})
        self.assertEqual(payload["objects"][0]["T_pelvis_object"], np.eye(4).tolist())
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
