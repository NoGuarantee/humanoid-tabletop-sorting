"""In-memory two-camera pose estimator. Its only scene inputs are RGB, Z, K and TF."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import fit_rgbd_templates as fit
from .full_pose_nn import AXIAL, OBJECTS, SCALE_M, features
from ..rgbd_labels import POOL_IDENTITIES, validate_detector_labels
from .scene_geometry import estimate_scene_geometry
from .rgbd_free_space import depth_view, rerank
from .rgbd_identity import IdentityHead
from .multiview_association import association, descriptors
from .rgbd_template_refinement import radial_profile, stable_tube
from .tube_end_ranker import TubeEndRanker
from .yolo_pose_head import SegmentationFeatures, YoloPoseHead, pair_features

from .detection import Detection


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PoseBackend:
    @staticmethod
    def pool_categories(pool):
        return POOL_IDENTITIES[pool]

    def __init__(self, root, *, model_poses=True):
        root = Path(root)
        bundle = json.loads((root / "bundle.json").read_text())
        expected_files = {"shape/catalog.json", "identity.pt", "pose.pt", "tube.pt", "yolo.pt"}
        if bundle["sha256"].keys() not in (expected_files, expected_files | {"box.pt"}):
            raise ValueError("invalid bundle manifest")
        required = bundle["sha256"] if model_poses else {"yolo.pt": bundle["sha256"]["yolo.pt"]}
        for filename, expected in required.items():
            if checksum(root / filename) != expected:
                raise ValueError(f"checksum mismatch for {filename}")
        from ultralytics import YOLO

        self.detector = YOLO(str(root / "yolo.pt"))
        validate_detector_labels(self.detector.names)
        self.hooks = SegmentationFeatures(self.detector) if model_poses else None
        if model_poses:
            self._load_pose_models(root, bundle)

    def _load_pose_models(self, root, bundle):
        self.index = json.loads((root / "shape" / "catalog.json").read_text())
        if self.index["objects"].keys() != set(OBJECTS):
            raise ValueError("unexpected model catalog names")
        self.catalog = {}
        fit.setup(root / "shape", self.catalog)
        self.calibrated = self.index["calibrated"]
        identity = torch.load(root / "identity.pt", map_location="cpu", weights_only=True)
        pose = torch.load(root / "pose.pt", map_location="cpu", weights_only=True)
        if any(tuple(saved["objects"]) != OBJECTS
               or saved["yolo_weights_sha256"] != bundle["sha256"]["yolo.pt"] for saved in (identity, pose)):
            raise ValueError("incompatible model and YOLO feature weights")
        self.channels = identity["channels"]
        self.identity = IdentityHead(self.channels).eval()
        self.identity.load_state_dict(identity["state"])
        self.pose = YoloPoseHead(self.channels).eval()
        self.pose.load_state_dict(pose["state_dict"])
        tube = torch.load(root / "tube.pt", map_location="cpu", weights_only=True)
        if tube["point_catalog_sha256"] != self.index["geometry_source_catalog_sha256"]:
            raise ValueError("tube head was trained with different geometry")
        state = tube["state"]
        self.tube = TubeEndRanker(state["lengths"].tolist(), state["radii"].tolist()).eval()
        self.tube.load_state_dict(state)
        self.profiles = {name: radial_profile(*self.catalog[name][:2]) for name in ("falcon50", "falcon15")}
        self.box = None
        if "box.pt" in bundle["sha256"]:
            from .paired_box_identity import NAMES, PairedBoxIdentity

            box = torch.load(root / "box.pt", map_location="cpu", weights_only=True)
            if (tuple(box["names"]) != NAMES or box["channels"] != self.channels
                    or box["yolo_weights_sha256"] != bundle["sha256"]["yolo.pt"]):
                raise ValueError("incompatible box classifier")
            self.box = PairedBoxIdentity(self.channels).eval()
            self.box.load_state_dict(box["state"])
            self.box_names = NAMES

    def segment(self, sensor):
        if self.hooks is not None:
            self.hooks.shape = self.hooks.map = None
        pred = self.detector.predict(np.ascontiguousarray(sensor["rgb"][..., ::-1]), device="cpu",
                                     imgsz=640, conf=.25, iou=.7, retina_masks=True, verbose=False)[0]
        raster = np.zeros(sensor["depth"].shape, dtype=np.uint16)
        objects = []
        if pred.masks is not None:
            masks = pred.masks.data.cpu().numpy().astype(bool)
            classes = pred.boxes.cls.cpu().numpy().astype(int)
            confidence = pred.boxes.conf.cpu().numpy()
            for i in np.argsort(-confidence, kind="stable"):
                ident = len(objects) + 1
                raster[masks[i] & (raster == 0)] = ident
                objects.append(dict(instance_id=ident, class_id=int(classes[i]) + 1,
                                    confidence=float(confidence[i])))
        crops = self.hooks.crops(raster) if self.hooks is not None else {}
        return raster, objects, crops

    def scene_geometry(self, sensors, *, segmentation=None, tray_of_color=None):
        """Table plane and tray openings, reusing segmentation when provided."""
        if segmentation is None:
            segmentation = [self.segment(sensor) if sensor is not None else None for sensor in sensors]
        return estimate_scene_geometry(sensors, segmentation, tray_of_color)

    def estimate(self, sensors, pool, segmentation, plane):
        """sensors are ordered head/torso; a missing slot is None."""
        masks, objects, crops, desc = [[], [], [], []]
        for index, sensor in enumerate(sensors):
            if sensor is None:
                masks.append(None)
                objects.append({})
                crops.append({})
                desc.append([])
                continue
            raster, proposals, neck = segmentation[index]
            proposals = [p for p in proposals if p["class_id"] in POOL_IDENTITIES[pool]]
            masks.append(raster)
            objects.append({p["instance_id"]: p for p in proposals})
            crops.append(neck)
            desc.append(descriptors(sensor["rgb"], sensor["depth"], sensor["K"], sensor["T"], raster, proposals))
        groups = association(*desc)
        for slot in range(2):
            used = {group[slot] for group in groups}
            groups.extend((i, None) if slot == 0 else (None, i) for i in objects[slot] if i not in used)
        tasks = []
        for ids in groups:
            refs = [objects[j][i] for j, i in enumerate(ids) if i is not None]
            category = refs[0]["class_id"]
            choices = POOL_IDENTITIES[pool][category]
            obs = [None if i is None else dict(sensors[j], mask=masks[j] == i) for j, i in enumerate(ids)]
            value = features(obs, OBJECTS[choices[0]])
            if value is None:
                continue
            image, extra, anchor = value
            neck = pair_features(obs, crops, ids, extra, self.channels)
            measured = [None if o is None else fit.measure(o["depth"], o["mask"], o["K"], o["T"],
                                                           self.calibrated) for o in obs]
            cloud = np.concatenate([m for m in measured if m is not None])
            if len(cloud) < 30:
                continue
            clipped = [o is not None and self.clipped(o["mask"]) for o in obs]
            tasks.append((ids, refs, category, choices, image, extra, neck, anchor, measured, cloud, clipped))
        if not tasks:
            return []
        guesses = self.guesses(tasks)
        depth_views = [depth_view(sensor) for sensor in sensors if sensor is not None]
        detections = [self.fit_pose(task, hypotheses, plane, depth_views) for task, hypotheses in zip(tasks, guesses)]
        return detections

    @staticmethod
    def clipped(mask):
        return bool(mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any())

    @torch.inference_mode()
    def guesses(self, tasks):
        image, extra, neck = [torch.from_numpy(np.stack([t[i] for t in tasks])).float() for i in (4, 5, 6)]
        logits, _ = self.identity(image, extra, neck)
        box_logits = None if self.box is None else self.box(image, extra, neck)[0]
        rows, labels, probabilities = [], [], []
        for i, task in enumerate(tasks):
            choices = list(task[3])
            p = logits[i, choices].softmax(0).numpy()
            if box_logits is not None and task[2] == 3:
                indices = [self.box_names.index(OBJECTS[label]) for label in choices]
                specialized = box_logits[i, indices].softmax(0).numpy()
                p = specialized if specialized.max() >= .90 else (p + specialized) / 2
            confident = .90 if box_logits is not None and task[2] == 3 else .98
            selected = np.argsort(-p)[:1 if p.max() >= confident else 2]
            for j in selected:
                rows.append(i)
                labels.append(choices[j])
                probabilities.append(float(p[j]))
        values = extra[rows].clone()
        values[:, 8:8 + len(OBJECTS)] = 0.
        values[torch.arange(len(rows)), 8 + torch.tensor(labels)] = 1.
        positions, rotations = self.pose(image[rows], values, neck[rows])
        guesses = [[] for _ in tasks]
        for i, label, probability, xyz, rotation in zip(rows, labels, probabilities, positions.numpy(), rotations.numpy()):
            t = np.eye(4)
            t[:3, :3] = rotation
            t[:3, 3] = tasks[i][7] + SCALE_M * xyz
            guesses[i].append((OBJECTS[label], t, probability))
        return guesses

    def fit_pose(self, task, hypotheses, plane, depth_views):
        ids, refs, category, _, image, extra, _, anchor, measured, cloud, clipped = task
        name, pose, info, options = fit.register_identities(hypotheses, cloud, plane, self.catalog)
        residual = float(info["trimmed_rmse_m"])
        orientation = "axis" if name in AXIAL else "full"
        bounds = self.catalog[name][5:7]
        low, high = bounds
        local_center = (low + high) / 2
        if category == 3:
            meta = self.index["objects"][name]
            orientation = "c" + str(meta["box_order"])
            local_center = np.asarray(meta["box_center_m"])
        uncertain = False
        points = self.catalog[name][3]
        if name in self.profiles:
            clouds = measured
            proposals, evidence = stable_tube(self.tube, self.profiles[name], bounds, pose, clouds,
                                              image, extra, anchor, name, clipped)
            pose = proposals[0]
            uncertain = evidence.get("tube_needs_reacquire", True)
            same_end = [option for option in options if option[2][:3, 2] @ pose[:3, 2] > 0.]
            candidate, candidate_residual = rerank(points, pose, info["combined_loss"], residual, same_end, depth_views)
            if candidate is not pose:
                proposals, evidence = stable_tube(self.tube, self.profiles[name], bounds, candidate, clouds,
                                                  image, extra, anchor, name, clipped)
                if proposals[0][:3, 2] @ pose[:3, 2] > 0. and not evidence.get("tube_needs_reacquire", True):
                    pose, residual, uncertain = proposals[0], candidate_residual, False
            local_center = np.array([0., 0., local_center[2]])
        else:
            pose, residual = rerank(points, pose, info["combined_loss"], residual, options, depth_views)
        views = sum(m is not None and len(m) >= 30 for m in measured)
        rotation = pose[:3, :3]
        center = rotation @ local_center + pose[:3, 3]
        valid = (np.isfinite(pose).all() and residual <= .012 and not uncertain
                 and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                 and abs(np.linalg.det(rotation) - 1.) <= 1e-5)
        return Detection(name, category, pose, center, high - low,
                         max(p["confidence"] for p in refs), residual, views, orientation,
                         bool(valid),
                         all(flag for flag, m in zip(clipped, measured) if m is not None and len(m) >= 30), points=len(cloud), instances=ids)
