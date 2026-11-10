"""cuRobo backend for MoveTcp. Owner: motion.

Active joints: body_yaw and both arms. Both TCP frames are goals at once: the arm that was asked to move goes to
the target, the other keeps the pose it has now. cuRobo quaternions are wxyz; ROS poses are xyzw.

The solver is built on first use. Kernel compilation is cached on disk by tools/curobo_warmup.py during setup.sh,
so a fresh container does not pay the cold Warp compile inside the task.
"""
import copy
import math
import time
from types import SimpleNamespace

from .curobo_robot import load_robot_cfg
from .robot import (
    ARM_JOINTS, CUBOID_CACHE, JOINTS, TCP_LINK, VERTICAL, clearance_z, fingertip, footprint_gap, on_the_way,
    plan_limits, ready_pose, sphere_cuboid_overlap, spheres_hit, tilt, world_scene)


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
CARRY_TURN = 0.35  # rad: the most a carry's joint path may turn the hand from the held orientation
CARRY_TURN_CHECKS = 24  # waypoints of a carry path whose TCP orientation is checked
NEAR_CUBOID = 0.01  # m: a sphere this close to a tray counts as in it, about cuRobo's collision activation distance
READY_LIFT = 0.08  # m: a hand left in the table or a tray goes this far straight up before the joint path to ready
UNDER_FINGERS = 0.05  # m: a straight descent that fails may go into objects whose footprint is this close to the
#                       goal fingertips. The grasped one can be measured off its true footprint, or a rim grasp sits
#                       past it; on the bench such descents failed with the start free.


class Unreachable(Exception):
    pass


def _with_z(target, z):
    """The same pose with a different TCP height. A ROS message and a namespace both have position and orientation."""
    o = target.orientation
    return SimpleNamespace(
        position=SimpleNamespace(x=float(target.position.x), y=float(target.position.y), z=float(z)),
        orientation=SimpleNamespace(x=float(o.x), y=float(o.y), z=float(o.z), w=float(o.w)))


def _xyzw(pose):
    o = pose.orientation
    return (float(o.x), float(o.y), float(o.z), float(o.w))


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
        self.table = None  # table plane (a, b, c, d) in pelvis. motion_node, from perception
        self.trays = ()  # tray tuples (robot.TRAY_SIZE) of this task that perception has seen. motion_node
        self.objects = ()  # detected neighbours, (id, x, y, z, yaw, sx, sy, sz, top_z). motion_node, from perception
        self.left_out = ()  # ids the last sync_world kept out of the world, the object under the fingertips
        self.free_yaw = False  # the last plan left the yaw of a top-down hand free: check its tilt, not the yaw
        self.carrying = False  # the hand holds an object: what the robot touches at the start is not an obstacle
        self._scene = None  # cuboid dict the collision world currently holds
        self.scene_note = ""  # what the last plan put in the world, for the move log
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
        self._build_self_check(cfg["kinematics"])
        t0 = time.perf_counter()
        # The cache is allocated here and never grown: update_world writes poses into it, and the CUDA graph keeps
        # those buffers. More cuboids than the cache raises inside cuRobo instead of rebuilding the planner.
        scene, _ = world_scene(self.table, self.trays, self.objects)
        self._scene = scene
        self.mp = MotionPlanner(MotionPlannerCfg.create(
            robot=copy.deepcopy(cfg), scene_model=scene, collision_cache={"cuboid": CUBOID_CACHE},
            use_cuda_graph=graphs != "off"))
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
        space can pass through the body, and the motors stall against it. When that fails and a neighbour stands
        on the way, one more try goes above the neighbour and down to the goal. A goal whose own pose is inside
        a cuboid stays unreachable: the fingers are not driven through it.
        `linear` is a Cartesian trajectory. The caller retimes either path to the 0.8 rad/s motor limit.
        dt is None only when the TCP is already at the goal and the waypoint is the current joints.
        """
        if self.mp is None:
            raise RuntimeError("curobo is not loaded")
        if mode not in ("direct", "linear", "ready"):
            raise RuntimeError(f"unknown mode {mode}")
        self.free_yaw = False
        if mode == "ready":
            # No objects: the hand may still hold one, or sit by the one it let go, and a start inside a cuboid
            # does not plan. main had no objects in the world for this move either.
            self.sync_world(None, objects=False)
            return self._plan_ready(arm, q_by_name)
        self.sync_world(target)
        held = self._held(q_by_name)
        if not held:
            return self._plan_in_world(arm, target, q_by_name, mode, pos_tol, rot_tol)
        result = self._without(held, lambda: self._plan_in_world(arm, target, q_by_name, mode, pos_tol, rot_tol))
        self.scene_note += f", without {', '.join(held)} (carried, the start touches them)"
        return result

    def _held(self, q_by_name):
        """Objects a carrying robot touches at the start: the one in the hand, and nothing to plan around.

        The look at the trays after the lift sees the object in the hand as one more object, a cuboid within a
        centimetre of the palm. With it in the world the straight carry across had no plan, and the joint-space
        path that replaced it went around the cartridge in the hand, rolling the wrist until it fell
        (sort3 seed 13051, 2026-11-10).
        """
        if not self.carrying:
            return []
        trays = set(self._tray_names())
        return [name for name in self._touching(q_by_name) if name not in trays and name != "table"]

    def _plan_in_world(self, arm, target, q_by_name, mode, pos_tol, rot_tol):
        """plan() after sync_world: `direct` with a detour, or `linear` without what the start touches."""
        if mode == "direct":
            if self.carrying:
                return self._plan_carry(arm, target, q_by_name, pos_tol, rot_tol)
            try:
                return self._plan_direct(arm, target, q_by_name, pos_tol, rot_tol)
            except Unreachable:
                detour = self._plan_detour(arm, target, q_by_name, pos_tol, rot_tol)
                if detour is None:
                    raise
                return detour
        goal, state = self._goal(arm, target, q_by_name)
        try:
            return self._plan_linear(arm, goal, state)
        except Unreachable as failed:
            touching = self._touching(q_by_name)
            trays = set(self._tray_names())
            tcp_z = self._tcp_z(arm, goal, state)
            rising = float(target.position.z) >= tcp_z + 0.01
            if not touching and float(target.position.z) <= tcp_z - 0.01:
                return self._descend_past_objects(arm, goal, state, target, failed)
            if not touching or not (rising or set(touching) <= trays):
                raise
            # cuRobo plans nothing from a start in collision. A hand that let go in a tray is inside a solid block
            # here, the trays being hollow in the scene; a hand at a low grasp is within a centimetre of the table
            # or of the object it holds. A move up, or out of a tray, goes away from them: plan it without them.
            result = self._without(touching, lambda: self._plan_linear(arm, goal, state))
            self.scene_note += f", planned without {', '.join(touching)} (the start touches them)"
            return result

    def _plan_carry(self, arm, target, q_by_name, pos_tol, rot_tol):
        """`direct` for a hand that holds an object: a path that keeps the hand's orientation, then the plain one.

        The plain joint path to an IK solution turns the wrist on the way. Carrying the cup to tray_right it was
        146 degrees off when the cup fell out, and the cartridge in the next run 170 (sort3 seed 13051,
        2026-11-10). A level path weighs the TCP orientation on every waypoint and frees only the position, so it
        may go around an obstacle but not roll the hand. Both of those carries had no level path with the grasp's
        yaw: the wrist runs out of range over the far tray. A top-down hand may turn about the vertical without
        tipping what it holds, so its last try leaves the yaw free (self.free_yaw, motion_node checks the tilt).
        A path that tips the hand more than CARRY_TURN is not taken.
        """
        notes = []
        tries = [False, True] if tilt(_xyzw(target)) <= VERTICAL else [False]
        for free_yaw in tries:
            try:
                plan, turn = self._plan_level(arm, target, q_by_name, free_yaw)
            except Unreachable as e:
                notes.append(f"level{' any yaw' if free_yaw else ''}: {e}")
                continue
            self.free_yaw = free_yaw
            self.scene_note += (f", level carry{' with a free yaw' if free_yaw else ''}, "
                                f"the hand {'tips' if free_yaw else 'turns'} {math.degrees(turn):.0f} deg at most")
            return plan
        try:
            plan = self._plan_direct(arm, target, q_by_name, pos_tol, rot_tol)
        except Unreachable:
            plan = self._plan_detour(arm, target, q_by_name, pos_tol, rot_tol)
            if plan is None:
                raise
        goal, state = self._goal(arm, target, q_by_name)
        turn = self._max_turn(arm, plan, goal, state, False)
        self.scene_note += f"; {'; '.join(notes)}; the joint path turns the hand {math.degrees(turn):.0f} deg"
        return plan

    def _plan_level(self, arm, target, q_by_name, free_yaw):
        """plan_pose with the TCP orientation held on every waypoint, or only its tilt when `free_yaw`.

        Returns (plan, the largest turn from the goal orientation, or the largest tilt when `free_yaw`).
        """
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        goal, state = self._goal(arm, target, q_by_name)
        link = TCP_LINK[arm]
        yaw_weight = 0.0 if free_yaw else 1.0
        self.mp.update_tool_pose_criteria({link: ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 1.0, 1.0, yaw_weight],
            non_terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, 1.0, 1.0, yaw_weight])})
        try:
            result = self.mp.plan_pose(goal, state)
        finally:
            self.mp.update_tool_pose_criteria({link: ToolPoseCriteria()})
        if result is None or not _flag(result.success, any_ok=True):
            raise Unreachable(f"no plan ({getattr(result, 'status', None)})")
        plan = self._trajectory(result)
        turn = self._max_turn(arm, plan, goal, state, free_yaw)
        if turn > CARRY_TURN:
            raise Unreachable(f"the path {'tips' if free_yaw else 'turns'} the hand {math.degrees(turn):.0f} deg")
        return plan, turn

    def _max_turn(self, arm, plan, goal, state, tilt_only):
        """The largest angle between the TCP orientation on `plan` and the goal's, over CARRY_TURN_CHECKS waypoints.

        With `tilt_only`, the largest angle of the TCP z axis from world z: the goal is a top-down hand.
        """
        import torch

        names, waypoints, _ = plan
        i = goal.tool_frames.index(TCP_LINK[arm])
        want = goal.quaternion[0, 0, i, 0]
        step = max(1, len(waypoints) // CARRY_TURN_CHECKS)
        worst = 0.0
        for q in waypoints[::step] + waypoints[-1:]:
            got = self.mp.compute_kinematics(self._chain_state(names, q, state)).tool_poses.quaternion[0, 0, i]
            if tilt_only:
                w, x, y, z = got.tolist()
                angle = tilt((x, y, z, w))
            else:
                dot = torch.abs(torch.dot(got.to(want.dtype), want)).clamp(max=1.0).item()
                angle = 2.0 * math.acos(dot)
            worst = max(worst, angle)
        return worst

    def _plan_direct(self, arm, target, q_by_name, pos_tol, rot_tol):
        """One collision-free path to `target`. No second attempt above the neighbours."""
        goal, state = self._goal(arm, target, q_by_name)
        if self._already(goal, state, arm, pos_tol, rot_tol):
            q = state.position.detach().reshape(-1).tolist()
            return list(self.joint_names), [q], None
        result = None
        solutions = []
        for _ in range(IK_TRIES):
            result = self.mp.ik_solver.solve_pose(
                goal, current_state=state, return_seeds=self.mp.ik_solver.config.num_seeds)
            # position_error is the worse of the two hands, and cuRobo stops at 5 mm. MoveTcp asks for one
            # hand, and the other hand is only held so it does not swing through the body.
            solutions = self._ranked(result, goal, arm, state, pos_tol, rot_tol)
            if solutions:
                break
        from curobo.types import JointState
        for solution in solutions[:MAX_PATHS]:
            goal_js = JointState.from_position(
                solution.detach().clone().reshape(1, -1), joint_names=self.joint_names)
            traj = self.mp.plan_cspace(goal_js, state)
            if traj is not None and _flag(traj.success, any_ok=True):
                return self._trajectory(traj)
        # Batch IK can miss the one pose that comes over a neighbour. The trajectory optimiser searches
        # configurations and the path together, so it still gets a try when every seed was in collision.
        traj = self.mp.plan_pose(goal, state)
        if traj is not None and _flag(traj.success, any_ok=True):
            return self._trajectory(traj)
        if not solutions:
            raise Unreachable(self._fail(result, goal, arm))
        start = [round(v, 3) for v in state.position.detach().reshape(-1).tolist()]
        raise Unreachable(
            f"no collision-free joint path to {min(len(solutions), MAX_PATHS)} IK solutions from {start}")

    def _plan_detour(self, arm, target, q_by_name, pos_tol, rot_tol):
        """Above the neighbours that stand between here and the goal, then down to it. None when that is no use.

        The waypoint keeps the goal's x, y and orientation, so the second leg is a descent onto the grasp.
        A pose that is itself inside a cuboid still fails here, and the caller reports the first error.
        """
        goal, state = self._goal(arm, target, q_by_name)
        z = self._clearance(arm, goal, state, target)
        if z is None:
            return None
        via = _with_z(target, z)
        try:
            names, way, dt = self._plan_direct(arm, via, q_by_name, pos_tol, rot_tol)
        except Unreachable as e:
            self.scene_note = f"{self.scene_note}; detour via failed: {e}"
            return None
        if dt is None or len(way) < 2:
            self.scene_note = f"{self.scene_note}; detour via did not move"
            return None
        q_via = dict(q_by_name)
        q_via.update(zip(names, way[-1]))
        try:
            names2, way2, _ = self._plan_direct(arm, target, q_via, pos_tol, rot_tol)
        except Unreachable as e:
            self.scene_note = f"{self.scene_note}; detour down failed: {e}"
            return None
        if set(names) != set(names2) or len(way2) < 2:
            return None
        if names != names2:
            index = {name: i for i, name in enumerate(names2)}
            way2 = [[q[index[name]] for name in names] for q in way2]
        self.scene_note = f"{self.scene_note}, detour"
        return names, way + way2[1:], dt

    def _clearance(self, arm, goal, state, target):
        """TCP z of the detour waypoint, or None when no neighbour stands on the straight line."""
        link = TCP_LINK[arm]
        i = goal.tool_frames.index(link)
        start = self.mp.compute_kinematics(state).tool_poses.position[0, 0, i].detach().tolist()
        end = (target.position.x, target.position.y, target.position.z)
        skipped = set(self.left_out)
        tops = [obj[8] for obj in self.objects
                if obj[0] not in skipped and on_the_way(obj, start, end)]
        return clearance_z(tops, start[2], end[2])

    def sync_world(self, target=None, objects=True):
        """Write the current objects into the collision world.

        With a target, the object under its fingertips stays out: the fingers have to enter it.
        Without one, every object stays. The joint raise checks that, before a goal exists.
        Rebuilding the planner for a new set of boxes costs seconds and recaptures the CUDA graph. The cuboid
        cache is fixed at load, and update_world only rewrites those tensors, which the captured graph already reads.
        """
        near = None
        if target is not None:
            position = target.position
            orientation = target.orientation
            near = fingertip((position.x, position.y, position.z),
                             (orientation.x, orientation.y, orientation.z, orientation.w))
        scene, left_out = world_scene(self.table, self.trays, self.objects if objects else (), near)
        self.left_out = tuple(left_out)
        n = len(scene["cuboid"]) - 1 - len(self.trays)
        note = ("" if self.table else "no table, ") + f"trays {[t[0] for t in self.trays]}, {n} objects"
        note += f", left out {list(left_out)}" if left_out else ""
        if scene == self._scene:
            self.scene_note = note
            return
        from curobo._src.geom.types import SceneCfg

        t0 = time.perf_counter()
        self.mp.update_world(SceneCfg.create(scene))
        self._scene = scene
        self.scene_note = f"{note}, world updated in {time.perf_counter() - t0:.2f} s"

    def segment_blocked(self, q_by_name, names, target):
        """True when this straight joint move would enter an object, and the start can already be planned from.

        Home is inside the thigh, so those first steps have to run: cuRobo will not plan from there. Once the
        arm is clear of itself, a later step that meets an object is refused and the MoveTcp plan goes around it.
        """
        cuboids = self._object_cuboids()
        if not cuboids or self.mp is None:
            return False
        if any(q_by_name.get(name) is None for name in list(self.joint_names) + list(names)):
            return False
        q = self._joints_tensor(q_by_name)
        if not bool(self._clear_of_self(q)[0].item()):
            return False
        start = [q_by_name[name] for name in names]
        for i in range(1, 9):
            t = i / 8.0
            pose = dict(q_by_name)
            for name, a, b in zip(names, start, target):
                pose[name] = a + (b - a) * t
            if spheres_hit(self._sphere_rows(pose), cuboids):
                return True
        return False

    def _object_cuboids(self):
        """Objects only. The table and the trays are what the joint raise was built to stay clear of."""
        if not self._scene:
            return {}
        skip = {"table", "no_table"} | set(self._tray_names())
        return {name: cub for name, cub in self._scene["cuboid"].items() if name not in skip}

    def _tray_names(self):
        """Trays the collision world holds now."""
        return [t[0] for t in self.trays if self._scene and t[0] in self._scene["cuboid"]]

    def _joints_tensor(self, q_by_name):
        import torch
        return torch.tensor([[float(q_by_name[name]) for name in self.joint_names]],
                            device="cuda", dtype=torch.float32)

    def _sphere_rows(self, q_by_name):
        from curobo.types import JointState
        state = JointState.from_position(self._joints_tensor(q_by_name), joint_names=self.joint_names)
        sph = self.mp.compute_kinematics(state).robot_spheres.detach().reshape(-1, 4)
        return [tuple(float(v) for v in row) for row in sph.cpu().tolist()]

    def _touching(self, q_by_name):
        """Cuboids of the world that a robot sphere is in or within NEAR_CUBOID of, at these joints."""
        if not self._scene or any(q_by_name.get(n) is None for n in self.joint_names):
            return []
        rows = [row for row in self._sphere_rows(q_by_name) if row[3] > 0.0]
        return [name for name, cub in self._scene["cuboid"].items() if name != "no_table"
                and any(sphere_cuboid_overlap((x, y, z), r, cub["dims"], cub["pose"]) > -NEAR_CUBOID
                        for x, y, z, r in rows)]

    def _under_fingers(self, target):
        """[(gap m, name)] of the objects in the world by the goal fingertips, nearest first."""
        p, o = target.position, target.orientation
        tip = fingertip((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        in_world = self._scene["cuboid"] if self._scene else {}
        return sorted((footprint_gap(obj, tip, self.table), obj[0]) for obj in self.objects
                      if obj[0] in in_world and obj[0] not in self.left_out)

    def _descend_past_objects(self, arm, goal, state, target, failed):
        """A straight descent from a free start failed: try it without the objects right under the fingertips."""
        near = self._under_fingers(target)
        self.scene_note += ", by the fingertips: " + (
            ", ".join(f"{name} {gap * 1000:+.0f} mm" for gap, name in near[:3]) or "no objects")
        names = [name for gap, name in near if gap <= UNDER_FINGERS]
        if not names:
            raise failed
        result = self._without(names, lambda: self._plan_linear(arm, goal, state))
        self.scene_note += f", descended without {', '.join(names)}"
        return result

    def _tcp_z(self, arm, goal, state):
        """Height of this arm's TCP in pelvis at `state`."""
        i = goal.tool_frames.index(TCP_LINK[arm])
        return float(self.mp.compute_kinematics(state).tool_poses.position[0, 0, i, 2].item())

    def start_report(self, q_by_name):
        """Why a start may not plan: self-collision overlap, and the nearest robot link to each cuboid. One line.

        Overlaps are in mm, positive inside. cuRobo already counts a sphere within about a centimetre as touching.
        """
        if self.mp is None or not self._scene or any(q_by_name.get(n) is None for n in self.joint_names):
            return "no report: planner or joints missing"
        q = self._joints_tensor(q_by_name)
        own = float(self._self_overlap(q)[0].item()) - START_CLEARANCE  # the plain cuRobo self-collision overlap
        rows = self._sphere_rows(q_by_name)
        kin = self._self_kin.config.kinematics_config
        index = kin.link_sphere_idx_map.tolist()
        links = {i: name for name, i in kin.link_name_to_idx_map.items()}
        near = []
        for name, cub in self._scene["cuboid"].items():
            if name == "no_table":
                continue
            best, link = -math.inf, "?"
            for i, (x, y, z, r) in enumerate(rows):
                if r <= 0.0:
                    continue
                d = sphere_cuboid_overlap((x, y, z), r, cub["dims"], cub["pose"])
                if d > best:
                    best, link = d, links.get(index[i], "?") if i < len(index) else "?"
            near.append((best, name, link))
        near.sort(reverse=True)
        parts = [f"{name} {best * 1e3:+.0f} mm ({link})" for best, name, link in near[:3]]
        return f"self {own * 1e3:+.1f} mm; nearest: " + (", ".join(parts) if parts else "empty world")

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
        traj = self._ready_cspace(goal_js, state)
        if traj is not None:
            return self._trajectory(traj)
        lifted = self._ready_after_lift(arm, q_by_name, state, goal_js)
        if lifted is not None:
            return lifted
        start = [round(v, 3) for v in q.reshape(-1).tolist()]
        raise Unreachable(f"no collision-free joint path to the ready {arm} arm from {start}")

    def _ready_cspace(self, goal_js, state):
        """plan_cspace to ready, then once more without the trays. None when neither plans."""
        traj = self.mp.plan_cspace(goal_js, state)
        if (traj is None or not _flag(traj.success, any_ok=True)) and self._tray_names():
            # The trays are solid blocks here and hollow in the scene: a hand can rest inside one (the left hand
            # 0.8 cm into tray_center on 2026-11-07), and then no state of that robot is free. Leave without them.
            traj = self._without(self._tray_names(), lambda: self.mp.plan_cspace(goal_js, state))
            if traj is not None and _flag(traj.success, any_ok=True):
                self.ready_note = "planned without the trays"
        return traj if traj is not None and _flag(traj.success, any_ok=True) else None

    def _ready_after_lift(self, arm, q_by_name, state, goal_js):
        """READY_LIFT straight up out of the cuboids the start touches, then the joint path to ready. None if no use.

        The robot is not reset between layouts, and the next one can start with a hand lying on the table: the left
        palm 2.7 cm into the table and 5.1 cm into tray_center, and neither arm planned to ready, so the layout scored
        nothing (eval --layouts 3, 2026-11-09). Switching the table off for the whole joint path could sweep the arm
        through it; only the vertical leg goes without it, as a linear lift does.
        """
        touching = self._touching(q_by_name)
        if not touching:
            return None
        goal = self.mp.compute_kinematics(state).tool_poses.clone().as_goal()
        i = goal.tool_frames.index(TCP_LINK[arm])
        goal.position[0, 0, i, 0, 2] += READY_LIFT
        try:
            names, up, dt = self._without(touching, lambda: self._plan_linear(arm, goal, state))
        except Unreachable:
            return None
        traj = self._ready_cspace(goal_js, self._chain_state(names, up[-1], state))
        if traj is None:
            return None
        rest_names, rest, _ = self._trajectory(traj)
        if rest_names != names:  # both come from the same motion generator; map by name all the same
            by_name = [dict(zip(rest_names, w)) for w in rest]
            rest = [[w.get(n, up[-1][k]) for k, n in enumerate(names)] for w in by_name]
        self.ready_note = ", ".join(filter(None, (
            f"lifted {READY_LIFT * 100:.0f} cm out of {', '.join(touching)} first", self.ready_note)))
        return names, up + rest, dt

    def _without(self, names, plan):
        """Run `plan` with these cuboids switched off in the collision world."""
        checker = self.mp.scene_collision_checker
        try:
            for name in names:
                checker.enable_obstacle(name, False)
            self.mp.graph_planner.reset_buffer()  # it keeps node feasibility from the old scene
            return plan()
        finally:
            for name in names:
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
