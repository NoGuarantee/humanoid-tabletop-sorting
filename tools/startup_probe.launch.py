"""The launch of run.sh with brain swapped for tools/startup_probe.py. Usage is there."""
import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    path = os.path.join(get_package_share_directory("hack_bringup"), "launch", "solution.launch.py")
    spec = importlib.util.spec_from_file_location("solution_launch", path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)
    keep = [e for e in solution.generate_launch_description().entities
            if not (isinstance(e, Node) and e.node_executable == "brain_node")]
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup_probe.py")
    return LaunchDescription([*keep, ExecuteProcess(cmd=["python3", "-u", probe], output="screen",
                                                    on_exit=Shutdown(reason="probe exited"))])
