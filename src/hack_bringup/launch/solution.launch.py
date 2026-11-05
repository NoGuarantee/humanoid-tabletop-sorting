"""All solution nodes, sim time everywhere. Started by run.sh."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    common = {"use_sim_time": True}
    rgbd = {
        "rgbd_model_dir": ParameterValue(LaunchConfiguration("rgbd_model_dir"), value_type=str),
        "rgbd_enable_model_poses": ParameterValue(LaunchConfiguration("rgbd_enable_model_poses"), value_type=bool),
    }
    return LaunchDescription([
        DeclareLaunchArgument("rgbd_model_dir", default_value=EnvironmentVariable("NSSR_RGBD_MODEL_DIR", default_value="")),
        DeclareLaunchArgument("rgbd_enable_model_poses", default_value="false"),
        Node(package="hack_motion", executable="motion_node", name="hack_motion",
             parameters=[common], output="screen"),
        Node(package="hack_perception", executable="perception_node", name="hack_perception",
             parameters=[common, rgbd], output="screen"),
        Node(package="hack_brain", executable="brain_node", name="hack_brain",
             parameters=[common], output="screen",
             on_exit=Shutdown(reason="brain exited")),  # brain owns /sim/task/done; without it nothing is left to do
    ])
