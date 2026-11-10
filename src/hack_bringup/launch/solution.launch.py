"""All solution nodes, sim time everywhere. Started by run.sh."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    common = {"use_sim_time": True}
    rgbd = {
        "serialized_images": ParameterValue(LaunchConfiguration("serialized_images"), value_type=bool),
        "perceive_frame_wait_s": ParameterValue(LaunchConfiguration("perceive_frame_wait_s"), value_type=float),
        "rgbd_model_dir": ParameterValue(LaunchConfiguration("rgbd_model_dir"), value_type=str),
        "rgbd_enable_model_poses": ParameterValue(LaunchConfiguration("rgbd_enable_model_poses"), value_type=bool),
        "rgbd_device": ParameterValue(LaunchConfiguration("rgbd_device"), value_type=str),
        "rgbd_on_demand": ParameterValue(LaunchConfiguration("rgbd_on_demand"), value_type=bool),
        "rgbd_model_names": ParameterValue(LaunchConfiguration("rgbd_model_names"), value_type=str),
        "rgbd_torch_threads": ParameterValue(LaunchConfiguration("rgbd_torch_threads"), value_type=int),
        "rgbd_pool": ParameterValue(LaunchConfiguration("rgbd_pool"), value_type=int),
    }
    return LaunchDescription([
        DeclareLaunchArgument("serialized_images", default_value="true"),
        DeclareLaunchArgument("perceive_frame_wait_s", default_value="0.75"),
        DeclareLaunchArgument("rgbd_model_dir", default_value=EnvironmentVariable("NSSR_RGBD_MODEL_DIR", default_value="")),
        DeclareLaunchArgument("recorded_grasps", default_value=EnvironmentVariable("NSSR_RECORDED_GRASPS", default_value="false")),
        DeclareLaunchArgument("rgbd_enable_model_poses", default_value=EnvironmentVariable("NSSR_RGBD_ENABLE_MODEL_POSES", default_value=LaunchConfiguration("recorded_grasps"))),
        DeclareLaunchArgument("rgbd_device", default_value=EnvironmentVariable("NSSR_RGBD_DEVICE", default_value="cpu")),
        DeclareLaunchArgument("rgbd_on_demand", default_value=EnvironmentVariable("NSSR_RGBD_ON_DEMAND", default_value="false")),
        DeclareLaunchArgument("rgbd_model_names", default_value=EnvironmentVariable("NSSR_RGBD_MODEL_NAMES", default_value="")),
        DeclareLaunchArgument("rgbd_torch_threads", default_value=EnvironmentVariable("NSSR_RGBD_TORCH_THREADS", default_value="2")),
        DeclareLaunchArgument("rgbd_pool", default_value="0"),
        Node(package="hack_motion", executable="motion_node", name="hack_motion",
             parameters=[common], output="screen"),
        Node(package="hack_perception", executable="perception_node", name="hack_perception",
             parameters=[common, rgbd], output="screen"),
        Node(package="hack_brain", executable="brain_node", name="hack_brain",
             parameters=[common, {"recorded_grasps": ParameterValue(LaunchConfiguration("recorded_grasps"), value_type=bool)}], output="screen",
             on_exit=Shutdown(reason="brain exited")),  # brain owns /sim/task/done; without it nothing is left to do
    ])
