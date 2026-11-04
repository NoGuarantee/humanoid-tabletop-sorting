"""cuRobo backend for MoveTcp. Owner: motion.

Active joints: body_yaw and both arms. Both TCP frames are goals at once: the arm that was asked to move goes to
the target, the other keeps the pose it has now. cuRobo quaternions are wxyz; ROS poses are xyzw.

The solver is built on first use. Kernel compilation is cached on disk by tools/curobo_warmup.py during setup.sh,
so a fresh container does not pay the cold Warp compile inside the task.
"""
import copy
import time

from .curobo_robot import load_robot_cfg
from .robot import JOINTS, TCP_LINK


LOCK_TOL = 0.02  # rad
START_MARGIN = 0.02  # rad inside the limits for the start state: a joint held at its stop reads slightly past it
LIMIT_MARGIN = 0.1  # rad from a limit: IK solutions closer than this are the last choice
MAX_PATHS = 4  # IK solutions to try a joint path to; a failed try costs about 1 s
SAME_SOLUTION = 0.3  # rad, joint-space distance: closer solutions fail the same way, so one try covers them


class Unreachable(Exception):
    pass


def _flag(success, any_ok=False):
    if isinstance(success, bool):
        return success
    flat = success.detach().reshape(-1)
    if flat.numel() == 0:
        return False
    return bool(flat.any().item()) if any_ok else bool(flat[0].item())


def _first(tensor):
    if tensor is None:
        return None
    return float(tensor.detach().reshape(-1)[0].item())


class CuroboPlanner:
    def __init__(self, urdf_path, cfg_path):
        self.urdf_path = urdf_path
        self.cfg_path = cfg_path
        with open(cfg_path) as f:
            import yaml
            data = yaml.safe_load(f)
        kin = data["kinematics"]
        locked = set((kin.get("lock_joints") or {}).keys())
        cspace = kin["cspace"]
        defaults = dict(zip(cspace["joint_names"], cspace["default_joint_position"]))
        # Home has the forearms inside the thighs. This pose (arms 0.2 rad out) is collision-free, and motion
        # moves here by joints before the first plan. body_yaw is left where the task put it.
        self.ready = {name: float(defaults[name]) for name in cspace["joint_names"]
                      if name not in locked and name != "body_yaw_joint"}
        self.mp = None
        self.joint_names = []
        self.locks = {}  # lock_joints the current planner was built with

    def load(self, lock_positions, graphs="reset"):
        """Create the motion planner. Returns a one-line description.

        cuRobo bakes lock_joints (legs, head) into fixed transforms. A later call with the same locks does
        nothing; if one moved by more than LOCK_TOL (brain tilts the head to look at the table), the planner is
        built again so the head spheres are where the head is. The kernels are cached, so that takes a few s.

        graphs: "reset" captures CUDA graphs and re-captures them when the problem shape changes (IK seeds,
        plan_cspace after plan_pose); "off" runs without graphs. cuRobo ships with graphs on and reset off, and
        then the first call of a different shape raises "CUDA graph reset is not available".
        """
        if self.mp is not None:
            moved = [name for name, value in self.locks.items()
                     if lock_positions and lock_positions.get(name) is not None
                     and abs(lock_positions[name] - value) > LOCK_TOL]
            if not moved:
                return "curobo already loaded"
            self.mp = None
            import gc
            import torch
            gc.collect()  # the old solvers hold CUDA graphs and buffers
            torch.cuda.empty_cache()
            reason = f"locked joints moved: {moved}; "
        else:
            reason = ""
        import curobo.runtime as runtime  # torch_util reads this copy, not curobo._src.runtime
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        runtime.cuda_graph_reset = graphs == "reset"
        cfg = load_robot_cfg(self.urdf_path, self.cfg_path, lock_positions)
        self.locks = {name: float(v) for name, v in (cfg["kinematics"].get("lock_joints") or {}).items()}
        t0 = time.perf_counter()
        self.mp = MotionPlanner(MotionPlannerCfg.create(
            robot=copy.deepcopy(cfg), use_cuda_graph=graphs != "off"))
        self.joint_names = list(self.mp.joint_names)
        torch.cuda.synchronize()
        return (f"{reason}curobo ready in {time.perf_counter() - t0:.1f} s, "
                f"{len(self.joint_names)} joints: {self.joint_names}")

    def warm_cache(self):
        """One IK and a planner warmup, so Warp and NVRTC write their kernel caches. Used by setup.sh."""
        import torch
        from curobo.types import JointState

        self.load(None)
        q = self.mp.ik_solver.sample_configs(1)
        if q.shape[0] == 0:  # no collision-free sample: the arms-out pose is the one the matrix was built in
            q = torch.zeros((1, len(self.joint_names)), device="cuda", dtype=torch.float32)
            for i, name in enumerate(self.joint_names):
                if name in self.ready:
                    q[0, i] = self.ready[name]
        else:
            q = q[:1]
        js = JointState.from_position(q, joint_names=self.joint_names)
        goal = self.mp.compute_kinematics(js).tool_poses.as_goal()
        t0 = time.perf_counter()
        result = self.mp.ik_solver.solve_pose(goal, current_state=js)
        torch.cuda.synchronize()
        ik_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.mp.warmup(enable_graph=True, num_warmup_iterations=3)
        torch.cuda.synchronize()
        warm_s = time.perf_counter() - t0
        ok = _flag(result.success)
        err = _first(result.position_error)
        cspace_ok = False
        if ok:
            # direct moves follow plan_cspace; compile those kernels here, not on the first task motion
            sol = result.solution.detach().reshape(-1, result.solution.shape[-1])[:1]
            goal_js = JointState.from_position(sol, joint_names=self.joint_names)
            traj = self.mp.plan_cspace(goal_js, js)
            torch.cuda.synchronize()
            cspace_ok = traj is not None and _flag(traj.success, any_ok=True)
        return {"ik_s": round(ik_s, 2), "warmup_s": round(warm_s, 2), "ik_ok": ok,
                "ik_pos_mm": None if err is None else round(err * 1e3, 3), "cspace_ok": cspace_ok}

    def plan(self, arm, target, q_by_name, mode, pos_tol=0.01, rot_tol=0.1):
        """Return (joint names, waypoints [N x dof], dt or None).

        `direct` is one IK solution, then a collision-free joint-space path to it. A straight line in joint
        space can pass through the body, and the motors stall against it.
        `linear` is a Cartesian trajectory. The caller retimes either path to the 0.8 rad/s motor limit.
        dt is None only when the TCP is already at the goal and the waypoint is the current joints.
        """
        if self.mp is None:
            raise RuntimeError("curobo is not loaded")
        if mode not in ("direct", "linear"):
            raise RuntimeError(f"unknown mode {mode}")
        goal, state = self._goal(arm, target, q_by_name)
        if mode == "direct":
            if self._already(goal, state, arm, pos_tol, rot_tol):
                q = state.position.detach().reshape(-1).tolist()
                return list(self.joint_names), [q], None
            result = self.mp.ik_solver.solve_pose(
                goal, current_state=state, return_seeds=self.mp.ik_solver.config.num_seeds)
            # position_error is the worse of the two hands, and cuRobo stops at 5 mm. MoveTcp asks for one
            # hand, and the other hand is only held so it does not swing through the body.
            solutions = self._ranked(result, goal, arm, state, pos_tol, rot_tol)
            if not solutions:
                raise Unreachable(self._fail(result, goal, arm))
            from curobo.types import JointState
            for solution in solutions[:MAX_PATHS]:
                goal_js = JointState.from_position(
                    solution.detach().clone().reshape(1, -1), joint_names=self.joint_names)
                traj = self.mp.plan_cspace(goal_js, state)
                if traj is not None and _flag(traj.success, any_ok=True):
                    return self._trajectory(traj)
            start = [round(v, 3) for v in state.position.detach().reshape(-1).tolist()]
            raise Unreachable(f"no collision-free joint path to {min(len(solutions), MAX_PATHS)} IK solutions from {start}")
        return self._plan_linear(arm, goal, state)

    def _plan_linear(self, arm, goal, state):
        """plan_pose with the moving TCP held on the line through the goal along one axis of the goal frame.

        cuRobo's linear criterion frees one goal-frame axis on the way and pulls the other two and the orientation
        to the goal. The axis is the one the move is mostly along: approach, lift and retreat are moves along the
        TCP z axis. A sideways start offset is pulled onto the line, not kept.
        """
        import torch
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        link = TCP_LINK[arm]
        i = goal.tool_frames.index(link)
        start = self.mp.compute_kinematics(state).tool_poses.position[0, 0, i]
        end = goal.position[0, 0, i, 0]
        w, x, y, z = goal.quaternion[0, 0, i, 0].tolist()
        rot = torch.tensor([  # goal frame -> pelvis
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], device=end.device, dtype=end.dtype)
        local = rot.T @ (end - start.to(end.dtype))
        axis = "xyz"[int(local.abs().argmax().item())]
        self.mp.update_tool_pose_criteria({link: ToolPoseCriteria.linear_motion(axis=axis)})
        try:
            result = self.mp.plan_pose(goal, state)
        finally:
            self.mp.update_tool_pose_criteria({link: ToolPoseCriteria()})
        if result is None or not _flag(result.success, any_ok=True):
            raise Unreachable(f"no straight collision-free path along the goal {axis} axis")
        return self._trajectory(result)

    def _trajectory(self, result):
        plan = result.get_interpolated_plan()
        pos = plan.position.detach()
        dof = pos.shape[-1]
        waypoints = pos.reshape(-1, dof).cpu().tolist()
        names = list(plan.joint_names) if getattr(plan, "joint_names", None) else list(self.joint_names)
        if len(names) != dof:
            raise RuntimeError(f"trajectory dof {dof} != {len(names)} names")
        if not waypoints:
            raise Unreachable("empty trajectory")
        cfg = getattr(getattr(self.mp, "trajopt_solver", None), "config", None)
        dt = float(getattr(cfg, "interpolation_dt", 0.05) or 0.05)
        return names, waypoints, dt

    def _already(self, goal, state, arm, pos_tol, rot_tol):
        """True when the commanded TCP is already inside the MoveTcp tolerance, so the joints must not move."""
        import torch

        link = TCP_LINK[arm]
        i = goal.tool_frames.index(link)
        pose = self.mp.compute_kinematics(state).tool_poses
        err = torch.linalg.norm(pose.position[0, 0, i] - goal.position[0, 0, i, 0]).item()
        dot = torch.abs(torch.dot(pose.quaternion[0, 0, i], goal.quaternion[0, 0, i, 0])).clamp(max=1.0).item()
        ang = 2.0 * torch.acos(torch.tensor(dot)).item()
        return err <= pos_tol and ang <= rot_tol

    def _ranked(self, result, goal, arm, state, pos_tol, rot_tol):
        """Joint vectors of the collision-free seeds whose commanded TCP is inside tolerance, best first.

        Closest to the current joints first: a far seed matches the TCP and still swings the other arm through the
        body. Seeds clear of the joint limits go ahead of the rest: the wrist range makes many top-down solutions
        sit on a stop, and a start state past a limit cannot be planned from. Near-duplicates are dropped.
        """
        import torch
        from curobo.types import JointState

        link = TCP_LINK[arm]
        i = goal.tool_frames.index(link)
        want_p = goal.position[0, 0, i, 0]
        want_q = goal.quaternion[0, 0, i, 0]
        current = state.position.detach().reshape(-1)
        sols = result.solution.detach().reshape(-1, result.solution.shape[-1])
        feas = result.feasible
        found = []
        for s in range(sols.shape[0]):
            if not isinstance(feas, bool) and not bool(feas.detach().reshape(-1)[s].item()):
                continue
            if isinstance(feas, bool) and not feas:
                continue
            js = JointState.from_position(sols[s].view(1, -1), joint_names=self.joint_names)
            pose = self.mp.compute_kinematics(js).tool_poses
            got_p = pose.position[0, 0, i]
            got_q = pose.quaternion[0, 0, i]
            err = torch.linalg.norm(got_p - want_p).item()
            dot = torch.abs(torch.dot(got_q, want_q)).clamp(max=1.0).item()
            ang = 2.0 * torch.acos(torch.tensor(dot)).item()
            dist = torch.linalg.norm(sols[s] - current).item()
            if err <= pos_tol and ang <= rot_tol:
                found.append((self._near_limit(sols[s].tolist()), dist, s))
        ranked = []
        for _, _, s in sorted(found):
            if all(torch.linalg.norm(sols[s] - other).item() > SAME_SOLUTION for other in ranked):
                ranked.append(sols[s])
        return ranked

    def _near_limit(self, q):
        for name, value in zip(self.joint_names, q):
            if name in JOINTS:
                _, _, lo, hi = JOINTS[name]
                if value < lo + LIMIT_MARGIN or value > hi - LIMIT_MARGIN:
                    return True
        return False

    def _inside_limits(self, name, value):
        if name not in JOINTS:
            return value
        _, _, lo, hi = JOINTS[name]
        return min(max(value, lo + START_MARGIN), hi - START_MARGIN)

    def _fail(self, result, goal, arm):
        """One line: best position and rotation error, and where the TCP actually ended up."""
        import torch
        from curobo.types import JointState

        pos = result.position_error.detach().reshape(-1)
        rot = result.rotation_error.detach().reshape(-1)
        feas = result.feasible if isinstance(result.feasible, bool) else bool(result.feasible.detach().reshape(-1)[0])
        link = TCP_LINK[arm]
        i = goal.tool_frames.index(link)
        want = goal.position[0, 0, i, 0].detach().cpu().tolist()
        got = "?"
        try:
            q = result.solution.detach().reshape(-1, result.solution.shape[-1])[0]
            js = JointState.from_position(q.view(1, -1), joint_names=self.joint_names)
            achieved = self.mp.compute_kinematics(js).tool_poses.position[0, 0, i]
            got = [round(v, 3) for v in achieved.detach().cpu().tolist()]
        except Exception as e:  # the shape of a failed result is not worth hiding the error numbers
            got = repr(e)
        return (f"ik best {pos[0].item() * 1e3:.0f} mm, {torch.rad2deg(rot[0]).item():.0f} deg, "
                f"feasible {feas}, {int(result.success.sum())}/{result.success.numel()} seeds; "
                f"want {[round(v, 3) for v in want]} got {got}")

    def _goal(self, arm, target, q_by_name):
        import torch
        from curobo.types import JointState

        missing = [name for name in self.joint_names if q_by_name.get(name) is None]
        if missing:
            raise RuntimeError(f"no /joint_states for {missing}")
        q = torch.tensor([[self._inside_limits(name, q_by_name[name]) for name in self.joint_names]],
                         device="cuda", dtype=torch.float32)
        state = JointState.from_position(q, joint_names=self.joint_names)
        goal = self.mp.compute_kinematics(state).tool_poses.clone().as_goal()
        link = TCP_LINK[arm]
        if link not in goal.tool_frames:
            raise RuntimeError(f"{link} is not a tool frame ({goal.tool_frames})")
        i = goal.tool_frames.index(link)
        pos = target.position
        quat = target.orientation  # ROS xyzw -> cuRobo wxyz
        dev, dtype = goal.position.device, goal.position.dtype
        goal.position[0, 0, i, 0] = torch.tensor([pos.x, pos.y, pos.z], device=dev, dtype=dtype)
        goal.quaternion[0, 0, i, 0] = torch.tensor(
            [quat.w, quat.x, quat.y, quat.z], device=dev, dtype=goal.quaternion.dtype)
        return goal, state
