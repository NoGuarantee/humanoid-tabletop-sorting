"""Blocking wrappers over the perception service and motion actions.

Called from the mission thread, never from a ROS callback: the node spins in a MultiThreadedExecutor in the main
thread, so waiting on a future here does not deadlock.
"""
import threading

from hack_interfaces.action import Hand, MoveJoints, MoveTcp
from hack_interfaces.srv import Perceive
from rclpy.action import ActionClient

TIMEOUT = 120.0  # real seconds; the sim runs slower than real time, so be generous


def _wait(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout)


class Clients:
    def __init__(self, node):
        self.node = node
        self.log = node.get_logger()
        self.perceive_cli = node.create_client(Perceive, "perception/perceive")
        self.move_tcp_cli = ActionClient(node, MoveTcp, "motion/move_tcp")
        self.move_joints_cli = ActionClient(node, MoveJoints, "motion/move_joints")
        self.hand_cli = ActionClient(node, Hand, "motion/hand")

    def wait_ready(self, timeout=60.0):
        ok = self.perceive_cli.wait_for_service(timeout_sec=timeout)
        for c in (self.move_tcp_cli, self.move_joints_cli, self.hand_cli):
            ok = c.wait_for_server(timeout_sec=timeout) and ok
        return ok

    def _action(self, client, goal, timeout=TIMEOUT):
        if not client.wait_for_server(timeout_sec=5.0):
            self.log.error(f"action server {client._action_name} is not available")
            return None
        fut = client.send_goal_async(goal)
        if not _wait(fut, 10.0) or not fut.result().accepted:
            return None
        res = fut.result().get_result_async()
        if not _wait(res, timeout):
            self.log.error(f"{client._action_name}: no result in {timeout} s")
            return None
        return res.result().result

    def perceive(self, camera="head", region="table"):
        """-> SceneObjects or None."""
        if not self.perceive_cli.wait_for_service(timeout_sec=5.0):
            return None
        fut = self.perceive_cli.call_async(Perceive.Request(camera=camera, region=region))
        if not _wait(fut, 30.0):
            return None
        r = fut.result()
        if not r.success:
            self.log.warn(f"perceive({camera}, {region}): {r.message}")
            return None
        return r.scene

    def move_tcp(self, arm, target, mode="direct", max_time=0.0, pos_tol=0.0):
        r = self._action(self.move_tcp_cli, MoveTcp.Goal(
            arm=arm, target=target, mode=mode, max_time=max_time, pos_tol=float(pos_tol)))
        if r is not None and not r.success:
            self.log.warn(f"move_tcp {arm} {mode}: {r.message}")
        return r is not None and r.success

    def move_joints(self, joints, positions, speed=0.0):
        r = self._action(self.move_joints_cli, MoveJoints.Goal(
            joints=list(joints), positions=[float(p) for p in positions], speed=float(speed)))
        if r is not None and not r.success:
            self.log.warn(f"move_joints {list(joints)}: {r.message}")
        return r is not None and r.success

    def hand(self, side, preset, force=0.0):
        """-> Hand.Result or None."""
        return self._action(self.hand_cli, Hand.Goal(hand=side, preset=preset, force=float(force)), timeout=30.0)
