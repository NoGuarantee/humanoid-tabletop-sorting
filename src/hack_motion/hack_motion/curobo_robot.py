"""Load the TienKung cuRobo config the way motion does at runtime. No ROS.

See robot/README.md: urdf_path points at a copy of the installed URDF with plan_limits, there are no meshes, and
lock_joints for the legs and the head are replaced with the current /joint_states before the solver is created.
"""
import tempfile
import xml.etree.ElementTree as ET

import yaml

from .robot import JOINTS, plan_limits


def _narrowed_urdf(urdf_path):
    """A temporary copy of the URDF with plan_limits. cuRobo takes its joint limits from the URDF only."""
    tree = ET.parse(urdf_path)
    for joint in tree.getroot().findall("joint"):
        limit = joint.find("limit")
        name = joint.get("name")
        if limit is None or name not in JOINTS:
            continue
        lo, hi = plan_limits(name)
        limit.set("lower", str(max(float(limit.get("lower")), lo)))
        limit.set("upper", str(min(float(limit.get("upper")), hi)))
    with tempfile.NamedTemporaryFile(prefix="tienkung_plan_", suffix=".urdf", delete=False) as f:
        tree.write(f)
    return f.name


def load_robot_cfg(urdf_path, cfg_path, lock_positions=None):
    """Return the robot dict for InverseKinematicsCfg / MotionPlannerCfg `robot=`.

    lock_positions: joint name -> radians. Only joints already listed in lock_joints are overwritten;
    anything missing stays at the home pose written in the yml.
    """
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    kin = data["kinematics"]
    kin["urdf_path"] = _narrowed_urdf(urdf_path)
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
