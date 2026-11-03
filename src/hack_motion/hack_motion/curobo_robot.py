"""Load the TienKung cuRobo config the way motion does at runtime. No ROS.

See robot/README.md: urdf_path points at the installed URDF, there are no meshes, and lock_joints for the
legs and the head are replaced with the current /joint_states before the solver is created.
"""
import yaml


def load_robot_cfg(urdf_path, cfg_path, lock_positions=None):
    """Return the robot dict for InverseKinematicsCfg / MotionPlannerCfg `robot=`.

    lock_positions: joint name -> radians. Only joints already listed in lock_joints are overwritten;
    anything missing stays at the home pose written in the yml.
    """
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    kin = data["kinematics"]
    kin["urdf_path"] = str(urdf_path)
    kin["asset_root_path"] = ""
    kin["load_meshes"] = False
    if lock_positions:
        locks = dict(kin.get("lock_joints") or {})
        for name in locks:
            value = lock_positions.get(name)
            if value is not None:
                locks[name] = float(value)
        kin["lock_joints"] = locks
    return data
