"""cuRobo backend for MoveTcp. Owner: motion.

Active joints: body_yaw and both arms. Both TCP frames are goals at once: the arm that was asked to move goes to
the target, the other keeps the pose it has now. cuRobo quaternions are wxyz; ROS poses are xyzw.

The solver is built on first use. Kernel compilation is cached on disk by tools/curobo_warmup.py during setup.sh,
so a fresh container does not pay the cold Warp compile inside the task.
"""
import copy
import math
import time

from .curobo_robot import load_robot_cfg
from .robot import ARM_JOINTS, JOINTS, TCP_LINK, plan_limits, ready_pose, table_scene


LOCK_TOL = 0.02  # rad
START_MARGIN = 0.02  # rad inside the limits for the start state: a joint held at its stop reads slightly past it
LIMIT_MARGIN = 0.1  # rad from a limit: IK solutions closer than this are the last choice
IK_TRIES = 4  # IK solves before calling a goal unreachable: beside the other, held hand 1-7 of the 32 seeds are free
MAX_PATHS = 4  # IK solutions to try a joint path to; a failed try costs about 1 s
SAME_SOLUTION = 0.3  # rad, joint-space distance: closer solutions fail the same way, so one try covers them
START_CLEARANCE = 0.002  # m past the self-collision buffer that a start state must have, or it is moved
START_CLEAR_STEPS = (0.01, 0.03, 0.1)  # rad, spreads of the samples around a start state in collision
START_CLEAR_SAMPLES = 512
START_CLEAR_MAX = 0.08  # rad: a farther jump is a different pose, and a held object comes off
# A 45° hand rises as two TCP-axis steps. Past 0.14 m that pair has no plan: the 0.15 m
# lift of the soap dish reports the start or the end in collision, while 0.12 m still plans.
LEAN_STEP = 0.12


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
        self.trays = ()  # trays of this task for the next build (motion_node, from /sim/task)
        self.world_trays = ()  # trays the current planner was built with
        self.start_moved = 0.0  # rad, how far the last plan moved its start state off the self-collision buffer
        self.ready_note = ""  # how the last `ready` plan was found, when not the plain way

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
            if not moved and self.world_trays == self.trays:
                return "curobo already loaded"
            self.mp = None
            import gc
            import torch
            gc.collect()  # the old solvers hold CUDA graphs and buffers
            torch.cuda.empty_cache()
            reason = f"locked joints moved: {moved}; trays {list(self.trays)}; "
        else:
            reason = ""
        import curobo.runtime as runtime  # torch_util reads this copy, not curobo._src.runtime
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        runtime.cuda_graph_reset = graphs == "reset"
        cfg = load_robot_cfg(self.urdf_path, self.cfg_path, lock_positions)
        self.locks = {name: float(v) for name, v in (cfg["kinematics"].get("lock_joints") or {}).items()}
        self._build_self_check(cfg["kinematics"])
        t0 = time.perf_counter()
        self.world_trays = self.trays
        self.mp = MotionPlanner(MotionPlannerCfg.create(
            robot=copy.deepcopy(cfg), scene_model=table_scene(self.trays), use_cuda_graph=graphs != "off"))
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
        if mode not in ("direct", "linear", "ready"):
            raise RuntimeError(f"unknown mode {mode}")
        if mode == "ready":
            return self._plan_ready(arm, q_by_name)
        goal, state = self._goal(arm, target, q_by_name)
        if mode == "direct":
            if self._already(goal, state, arm, pos_tol, rot_tol):
                q = state.position.detach().reshape(-1).tolist()
                return list(self.joint_names), [q], None
            for _ in range(IK_TRIES):
                result = self.mp.ik_solver.solve_pose(
                    goal, current_state=state, return_seeds=self.mp.ik_solver.config.num_seeds)
                # position_error is the worse of the two hands, and cuRobo stops at 5 mm. MoveTcp asks for one
                # hand, and the other hand is only held so it does not swing through the body.
                solutions = self._ranked(result, goal, arm, state, pos_tol, rot_tol)
                if solutions:
                    break
            if not solutions:
                raise Unreachable(self._fail(result, goal, arm))
            from curobo.types import JointState
            for solution in solutions[:MAX_PATHS]:
                goal_js = JointState.from_position(
                    solution.detach().clone().reshape(1, -1), joint_names=self.joint_names)
                traj = self.mp.plan_cspace(goal_js, state)
                if traj is not None and _flag(traj.success, any_ok=True):
                    return self._trajectory(traj)
            traj = self.mp.plan_pose(goal, state)  # free to end at any solution, not one fixed in advance
            if traj is not None and _flag(traj.success, any_ok=True):
                return self._trajectory(traj)
            start = [round(v, 3) for v in state.position.detach().reshape(-1).tolist()]
            raise Unreachable(f"no collision-free joint path to {min(len(solutions), MAX_PATHS)} IK solutions from {start}")
        return self._plan_linear(arm, goal, state)

    def _plan_ready(self, arm, q_by_name):
        """A collision-free joint path that puts `arm` and body_yaw back to robot.ready_pose.

        `direct` picks the IK solution nearest the current joints, so after a place, or after a pose-only park,
        the arm can sit by a stop (elbow_yaw -2.95, wrist_pitch 1.03 on 2026-11-07), and from there no pre-grasp
        plans. Every pick starts from fixed joints instead. The other arm keeps its joints.
        """
        import torch
        from curobo.types import JointState

        self.ready_note = ""
        missing = [name for name in self.joint_names if q_by_name.get(name) is None]
        if missing:
            raise RuntimeError(f"no /joint_states for {missing}")
        q = torch.tensor([[self._inside_limits(name, q_by_name[name]) for name in self.joint_names]],
                         device="cuda", dtype=torch.float32)
        q = self._clear_start(q)
        state = JointState.from_position(q, joint_names=self.joint_names)
        want = dict(zip(self.joint_names, q.reshape(-1).tolist()))
        want.update(zip(ARM_JOINTS[arm], ready_pose(arm)))
        if "body_yaw_joint" in want:
            want["body_yaw_joint"] = self.ready.get("body_yaw_joint", 0.0)
        if max(abs(want[n] - v) for n, v in zip(self.joint_names, q.reshape(-1).tolist())) < LOCK_TOL:
            return list(self.joint_names), [q.reshape(-1).tolist()], None
        goal_js = JointState.from_position(
            torch.tensor([[want[n] for n in self.joint_names]], device=q.device, dtype=q.dtype),
            joint_names=self.joint_names)
        traj = self.mp.plan_cspace(goal_js, state)
        if (traj is None or not _flag(traj.success, any_ok=True)) and self.world_trays:
            # The trays are solid blocks here and hollow in the scene: a hand can rest inside one (the left hand
            # 0.8 cm into tray_center on 2026-11-07), and then no state of that robot is free. Leave without them.
            traj = self._without_trays(lambda: self.mp.plan_cspace(goal_js, state))
            if traj is not None and _flag(traj.success, any_ok=True):
                self.ready_note = "planned without the trays"
        if traj is None or not _flag(traj.success, any_ok=True):
            start = [round(v, 3) for v in q.reshape(-1).tolist()]
            raise Unreachable(f"no collision-free joint path to the ready {arm} arm from {start}")
        return self._trajectory(traj)

    def _without_trays(self, plan):
        checker = self.mp.scene_collision_checker
        try:
            for name in self.world_trays:
                checker.enable_obstacle(name, False)
            self.mp.graph_planner.reset_buffer()  # it keeps node feasibility from the old scene
            return plan()
        finally:
            for name in self.world_trays:
                checker.enable_obstacle(name, True)
            self.mp.graph_planner.reset_buffer()

    def _plan_linear(self, arm, goal, state):
        """plan_pose with the moving TCP held on the line through the goal along one axis of the goal frame.

        cuRobo's linear criterion frees one goal-frame axis on the way and pulls the other two and the orientation
        to the goal. A move that lies along one TCP axis uses that axis. A world-vertical move of a leaned hand
        does not: it is two axis moves, the one that gains height first, so the path is not pulled down into the table.
        """
        import torch

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
        length = float(torch.linalg.norm(local))
        dominant = int(local.abs().argmax().item())
        # Longer than one leaned step: cut the straight move into pieces, each planned below.
        if length > LEAN_STEP and float(local[dominant].abs()) <= 0.85 * length:
            n = math.ceil(length / LEAN_STEP)
            names, waypoints, dt = None, [], None
            js = state
            for step_i in range(1, n + 1):
                alpha = step_i / n
                step = goal.clone()
                step.position[0, 0, i, 0] = start.to(end.dtype) + (end - start.to(end.dtype)) * alpha
                try:
                    names, part, dt = self._plan_linear(arm, step, js)
                except Unreachable as e:
                    raise Unreachable(f"at {alpha:.2f} of {length:.3f} m: {e}") from e
                waypoints.extend(part)
                js = self._chain_state(names, part[-1], js)
            return names, waypoints, dt
        # A leaned hand's world-vertical move is split across two TCP axes. Freeing only the larger one
        # pulls the path onto a line that, for the 45° pinch, points down into the table.
        if length < 1e-4 or float(local[dominant].abs()) > 0.85 * length:
            return self._linear_axis(link, goal, state, "xyz"[dominant])
        order = sorted(range(3), key=lambda k: -float(rot[2, k] * local[k]))  # gain height before sliding
        names, waypoints, dt = None, [], None
        here = start.to(end.dtype).clone()
        js = state
        for k in order:
            if abs(float(local[k])) < 1e-4:
                continue
            step = goal.clone()
            here = here + rot[:, k] * local[k]
            step.position[0, 0, i, 0] = here
            names, part, dt = self._linear_axis(link, step, js, "xyz"[k])
            waypoints.extend(part)
            js = self._chain_state(names, part[-1], js)
        return names, waypoints, dt

    def _chain_state(self, names, waypoint, prev):
        """Joint state at the end of one axis step, in the planner cspace.

        The interpolated plan also carries locked joints. Passing that wider vector back in makes the
        IK seed 26 wide while the solver is the 15-dof cspace, and the next plan_pose crashes.
        """
        import torch
        from curobo.types import JointState

        got = {name: float(value) for name, value in zip(names, waypoint)}
        prev_names = list(prev.joint_names) if getattr(prev, "joint_names", None) else list(self.joint_names)
        keep = {name: float(value) for name, value in zip(prev_names, prev.position.detach().reshape(-1).tolist())}
        q = [got.get(name, keep.get(name, 0.0)) for name in self.joint_names]
        return JointState.from_position(
            torch.tensor([q], device=prev.position.device, dtype=prev.position.dtype),
            joint_names=list(self.joint_names))

    def _linear_axis(self, link, goal, state, axis):
        """One Cartesian plan that may travel only along `axis` of the goal frame."""
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        self.mp.update_tool_pose_criteria({link: ToolPoseCriteria.linear_motion(axis=axis)})
        try:
            result = self.mp.plan_pose(goal, state)
        finally:
            self.mp.update_tool_pose_criteria({link: ToolPoseCriteria()})
        if result is None or not _flag(result.success, any_ok=True):
            raise Unreachable(f"no straight collision-free path along the goal {axis} axis; start {self._violated(state)}")
        return self._trajectory(result)

    def _violated(self, state):
        """The constraints a joint state breaks, as the graph planner's start check sees them, e.g. 'scene_collision 39'."""
        try:
            gp = self.mp.graph_planner
            buf = gp._max_act_buffer
            buf[:] = state.position.detach().reshape(1, 1, -1)
            cc = gp.feasibility_rollout.compute_metrics_from_action(buf).costs_and_constraints
            bad = []
            for group in (cc.constraints, cc.hybrid_costs_constraints):
                for name, value in zip(getattr(group, "names", []), getattr(group, "values", [])):
                    if value is not None and float(value[0].max().item()) > 0.0:
                        bad.append(f"{name} {float(value[0].max().item()):.3g}")
            return ", ".join(bad) or "free"
        except Exception as e:  # diagnostics only: never hide the planning error behind this one
            return f"? ({e!r})"

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
                lo, hi = plan_limits(name)
                if value < lo + LIMIT_MARGIN or value > hi - LIMIT_MARGIN:
                    return True
        return False

    def _inside_limits(self, name, value):
        if name not in JOINTS:
            return value
        lo, hi = plan_limits(name)
        return min(max(value, lo + START_MARGIN), hi - START_MARGIN)

    def _build_self_check(self, kinematics):
        from curobo._src.cost.cost_self_collision import SelfCollisionCost
        from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
        from curobo._src.robot.kinematics.kinematics import Kinematics
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

        kin_cfg = KinematicsCfg.from_data_dict(copy.deepcopy(kinematics))
        self._self_kin = Kinematics(kin_cfg)
        self._self_cost = SelfCollisionCost(SelfCollisionCostCfg(
            weight=1.0, store_pair_distance=True, self_collision_kin_config=kin_cfg.self_collision_config))

    def _self_overlap(self, q):
        """Max sphere overlap (m) per row of q [N x dof], after the start clearance is added.

        Positive means cuRobo will refuse to plan from that state. The cost already includes the
        self-collision buffer; the radius bump is the extra START_CLEARANCE.
        """
        from curobo.types import JointState

        n = q.shape[0]
        state = JointState.from_position(q, joint_names=self.joint_names)
        spheres = self._self_kin.compute_kinematics(state).robot_spheres.clone()
        spheres[..., 3] += START_CLEARANCE / 2.0
        self._self_cost.setup_batch_tensors(n, 1)
        self._self_cost.forward(spheres)
        return self._self_cost._pair_distance.view(n, -1).amax(dim=1)

    def _clear_of_self(self, q):
        """One bool per row of q [N x dof]: every sphere pair is START_CLEARANCE past the self-collision buffer."""
        return self._self_overlap(q) <= 0.0

    def _bounds(self, q):
        import torch
        lo = torch.tensor([self._inside_limits(name, -math.inf) for name in self.joint_names]).to(q)
        hi = torch.tensor([self._inside_limits(name, math.inf) for name in self.joint_names]).to(q)
        return lo, hi

    def _project_clear(self, q, lo, hi):
        """Walk q [1 x dof] out of the buffer along the overlap gradient.

        Takes the shortest step that is already clear. A shorter step that only reduces the overlap can use up
        START_CLEAR_MAX before the pose is clear, and then no step is left. None when nothing within the cap helps.
        """
        import torch

        eps = 0.02
        dof = q.shape[1]
        eye = torch.eye(dof, device=q.device, dtype=q.dtype)
        cur = q
        for _ in range(6):
            overlap = float(self._self_overlap(cur)[0].item())
            if overlap <= 0.0:
                return cur
            samples = torch.clamp(torch.cat([cur + eps * eye, cur - eps * eye], dim=0), min=lo, max=hi)
            pen = self._self_overlap(samples)
            grad = (pen[:dof] - pen[dof:]) / (2.0 * eps)
            norm = torch.linalg.norm(grad)
            if float(norm) < 1e-8:
                return None
            direction = grad / norm
            improving = None
            for length in (0.01, 0.02, 0.04, 0.08):
                trial = torch.clamp(cur - length * direction, min=lo, max=hi)
                if float(torch.linalg.norm(trial - q).item()) > START_CLEAR_MAX:
                    continue
                gap = float(self._self_overlap(trial)[0].item())
                if gap <= 0.0:
                    return trial  # the shortest step that is already clear; a shorter one may only nibble
                if gap < overlap and improving is None:
                    improving = trial
            if improving is None:
                return None
            cur = improving
        return cur if float(self._self_overlap(cur)[0].item()) <= 0.0 else None

    def _clear_start(self, q):
        """The nearest state to q [1 x dof] that can be planned from.

        cuRobo fails every plan from a start inside the self-collision buffer. IK ends on the edge of the buffer, and
        the drives stop short of the command, so the arm does end up inside, and then every goal fails until it moves.
        A random sample around that pose is often 0.2 rad away: the trajectory then starts with a jump and a held
        object comes off. The overlap gradient picks a step inside START_CLEAR_MAX, the shortest one that is clear.
        Returns q itself when it is clear, or when nothing close is: the plan then reports the failure.
        """
        import torch

        self.start_moved = 0.0
        if self._clear_of_self(q)[0]:
            return q
        lo, hi = self._bounds(q)
        projected = self._project_clear(q, lo, hi)
        if projected is not None and self._clear_of_self(projected)[0]:
            self.start_moved = float(torch.linalg.norm(projected - q).item())
            return projected
        rng = torch.Generator(device=q.device).manual_seed(0)  # the global generator seeds cuRobo
        best_q, best_d = None, START_CLEAR_MAX
        for spread in START_CLEAR_STEPS:
            noise = torch.randn((START_CLEAR_SAMPLES, q.shape[1]), device=q.device, generator=rng)
            samples = torch.clamp(q + spread * noise.to(q), min=lo, max=hi)
            distance = torch.linalg.norm(samples - q, dim=1)
            clear = self._clear_of_self(samples) & (distance <= START_CLEAR_MAX)
            if bool(clear.any()):
                picked = distance[clear]
                i = int(torch.arange(samples.shape[0], device=q.device)[clear][int(picked.argmin().item())].item())
                if float(distance[i]) < best_d:
                    best_q, best_d = samples[i], float(distance[i])
                    break
        if best_q is None:
            return q
        self.start_moved = best_d
        return best_q.reshape(1, -1)

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
        q = self._clear_start(q)
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
