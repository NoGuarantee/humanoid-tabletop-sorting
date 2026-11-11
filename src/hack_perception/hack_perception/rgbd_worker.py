"""Isolated inference with one pending job and one prepared exposure."""

from multiprocessing import get_context
import os
from queue import Empty, Full
from time import monotonic
from uuid import uuid4


def _serve(requests, results, model_dir, model_poses, torch_threads, device='cpu', robot_self_filter=False):
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        import torch

        torch.set_num_threads(torch_threads)
        from .rgbd.pose_backend import PoseBackend
        from .rgbd_response import bind_exposure, infer_response, prepare_scene
        from .rgbd_frames import CAMERAS

        backend = PoseBackend(model_dir, model_poses=model_poses, device=device)
        if robot_self_filter:
            from .rgbd_robot_filter import filter_sensor_packets, filter_provenance, load_robot_geometry
            robot_geometry=load_robot_geometry()
    except Exception as exc:
        results.put(("failed", None, repr(exc), None))
        return
    results.put(("ready", None, None, None))
    cache_key, sensors, prepared = None, None, None
    session_id = uuid4().hex
    while True:
        command = requests.get()
        if command is None:
            return
        kind, key, data = command
        start = monotonic()
        try:
            if kind == "scene":
                new_sensors, tray_of_color = data
                if robot_self_filter:
                    for sensor in new_sensors:
                        if sensor is not None and (sensor.get('stamp_ns')!=key[1]
                                or sensor.get('robot_joint_state',{}).get('generation')!=key[0]):
                            raise ValueError('robot_filter_exposure_epoch_mismatch')
                    new_sensors=filter_sensor_packets(new_sensors,robot_geometry)
                new_prepared = prepare_scene(backend, new_sensors, tray_of_color)
                cache_key, sensors, prepared = key, new_sensors, new_prepared
                value = prepared[1:]
            elif kind == "pose" and model_poses and prepared is not None and key == cache_key:
                if isinstance(data, dict):
                    value = infer_response(backend, sensors, key[1], data["pool"],
                                           prepared=prepared, names=data["names"])
                else:
                    value = infer_response(backend, sensors, key[1], data, prepared=prepared)
                if isinstance(value, dict):
                    bind_exposure(value, key, session_id,
                                  [camera for camera, sensor in zip(CAMERAS, sensors) if sensor is not None])
                    if robot_self_filter:
                        value['robot_self_filter']=filter_provenance(sensors)
            else:
                raise ValueError("pose request requires the prepared exposure")
        except Exception as exc:
            results.put(("job_failed", key, repr(exc), monotonic() - start))
        else:
            results.put((kind, key, value, monotonic() - start))


class PoseWorker:
    def __init__(self, model_dir, model_poses, torch_threads, device='cpu', robot_self_filter=False):
        context = get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=1)
        self.process = context.Process(target=_serve, args=(self.requests, self.results, model_dir,
                                                            model_poses, torch_threads, device, robot_self_filter), daemon=True)
        self.process.start()

    def submit(self, kind, key, data):
        try:
            self.requests.put_nowait((kind, key, data))
        except Full:
            return False
        return True

    def poll(self):
        try:
            return self.results.get_nowait()
        except Empty:
            return None

    def close(self):
        try:
            self.requests.put_nowait(None)
        except Full:
            pass
        self.process.join(timeout=.3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        for channel in (self.requests, self.results):
            channel.cancel_join_thread()
            channel.close()


def is_fresh(key, generation, now_ns, max_age_ns):
    epoch, stamp = key
    return epoch == generation and 0 <= now_ns - stamp <= max_age_ns
