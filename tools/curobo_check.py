#!/usr/bin/env python3
"""cuRobo first checks (docs/ANALYSIS.md, 7.3): import, JIT warm-up, IK and plan time, GPU and host memory.

Runs on a stock cuRobo robot (franka.yml by default), before the TienKung config exists. Start the simulator first
so the GPU budget is measured next to it. In a fresh container from the simulator image:

  bash tools/remote.sh dev --timeout 1800 -- 'bash tools/curobo_install.sh && python3 tools/curobo_check.py'

Prints a human-readable report and a final JSON line (prefix CUROBO_CHECK).
"""
import argparse
import json
import resource
import statistics
import subprocess
import time

T0 = time.perf_counter()


def now():
    return time.perf_counter()


def gpu_used_mib():
    """Whole-device memory in use (all processes, the simulator included), MiB."""
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20, total / 2**20


def rss_peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def ms(xs):
    return {"median_ms": round(statistics.median(xs) * 1e3, 1), "max_ms": round(max(xs) * 1e3, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="franka.yml")
    ap.add_argument("--n", type=int, default=20, help="timed IK solves and plans")
    ap.add_argument("--batch", type=int, default=64, help="batched IK size (grasp candidates)")
    args = ap.parse_args()
    out = {"robot": args.robot}

    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["gpu"] = torch.cuda.get_device_name(0)
    out["sm"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
    out["arch_list"] = torch.cuda.get_arch_list()
    used0, total = gpu_used_mib()
    out["gpu_total_mib"] = round(total)
    out["gpu_used_before_mib"] = round(used0)

    t = now()
    import curobo
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    out["curobo"] = getattr(curobo, "__version__", "?")
    out["import_s"] = round(now() - t, 2)

    # --- IK: cold first solve includes NVRTC compilation of the kernels ---
    t = now()
    ik = InverseKinematics(InverseKinematicsCfg.create(robot=args.robot, num_seeds=32,
                                                       self_collision_check=True))
    out["ik_create_s"] = round(now() - t, 2)
    q = ik.sample_configs(args.n + 1)

    def goal(i):  # a reachable target: FK of a sampled collision-free configuration
        js = JointState.from_position(q[i:i + 1], joint_names=ik.joint_names)
        return ik.compute_kinematics(js).tool_poses.as_goal()

    t = now()
    r = ik.solve_pose(goal(0))
    torch.cuda.synchronize()
    out["ik_first_solve_s"] = round(now() - t, 2)

    times, errs, ok = [], [], 0
    for i in range(1, args.n + 1):
        t = now()
        r = ik.solve_pose(goal(i))
        torch.cuda.synchronize()
        times.append(now() - t)
        ok += int(r.success.item())
        errs.append(r.position_error.item())
    out["ik_single"] = {**ms(times), "success": f"{ok}/{args.n}",
                        "pos_err_max_mm": round(max(errs) * 1e3, 3)}
    del ik
    torch.cuda.empty_cache()

    # --- batched IK, the grasp-candidate ranking case ---
    t = now()
    ikb = InverseKinematics(InverseKinematicsCfg.create(robot=args.robot, num_seeds=32, self_collision_check=True,
                                                        max_batch_size=args.batch))
    qb = ikb.sample_configs(args.batch)
    gb = ikb.compute_kinematics(JointState.from_position(qb, joint_names=ikb.joint_names)).tool_poses.as_goal()
    ikb.solve_pose(gb)
    torch.cuda.synchronize()
    out["ik_batch_create_and_first_s"] = round(now() - t, 2)
    times = []
    for _ in range(5):
        t = now()
        r = ikb.solve_pose(gb)
        torch.cuda.synchronize()
        times.append(now() - t)
    out["ik_batch"] = {**ms(times), "batch": args.batch, "success": f"{int(r.success.sum())}/{args.batch}"}
    del ikb
    torch.cuda.empty_cache()

    # --- motion planner: IK + graph + trajopt ---
    t = now()
    mp = MotionPlanner(MotionPlannerCfg.create(robot=args.robot, scene_model="collision_table.yml"))
    out["mp_create_s"] = round(now() - t, 2)
    t = now()
    mp.warmup(enable_graph=True, num_warmup_iterations=5)
    torch.cuda.synchronize()
    out["mp_warmup_s"] = round(now() - t, 2)

    start = JointState.from_position(mp.default_joint_state.position.unsqueeze(0), joint_names=mp.joint_names)
    qs = mp.ik_solver.sample_configs(args.n)
    times, ok, durations = [], 0, []
    for i in range(args.n):
        g = mp.compute_kinematics(JointState.from_position(qs[i:i + 1], joint_names=mp.joint_names))
        t = now()
        r = mp.plan_pose(g.tool_poses.as_goal(), start)
        torch.cuda.synchronize()
        times.append(now() - t)
        if r is not None and bool(r.success.any()):
            ok += 1
            durations.append(r.get_interpolated_plan().position.shape[-2] * mp.trajopt_solver.config.interpolation_dt)
    out["plan"] = {**ms(times), "success": f"{ok}/{args.n}",
                   "traj_duration_median_s": round(statistics.median(durations), 2) if durations else None}

    used1, _ = gpu_used_mib()
    out["gpu_used_after_mib"] = round(used1)
    out["gpu_ours_mib"] = round(used1 - used0)
    out["torch_max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
    out["rss_peak_mib"] = round(rss_peak_mib())
    out["total_s"] = round(now() - T0, 1)

    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        out["nvidia_smi"] = smi
    except Exception as e:  # noqa: BLE001
        out["nvidia_smi"] = repr(e)

    for k, v in out.items():
        print(f"{k:28s} {v}")
    print("CUROBO_CHECK " + json.dumps(out))


if __name__ == "__main__":
    main()
