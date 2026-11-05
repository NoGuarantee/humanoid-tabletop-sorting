"""All solution nodes, sim time everywhere. Started by run.sh."""
from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    common = {"use_sim_time": True}
    return LaunchDescription([
        Node(package="hack_motion", executable="motion_node", name="hack_motion",
             parameters=[common], output="screen"),
        Node(package="hack_perception", executable="perception_node", name="hack_perception",
             parameters=[common], output="screen"),
        Node(package="hack_brain", executable="brain_node", name="hack_brain",
             parameters=[common], output="screen",
             on_exit=Shutdown(reason="brain exited")),  # brain owns /sim/task/done; without it nothing is left to do
    ])
