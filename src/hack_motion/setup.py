from glob import glob

from setuptools import setup

package_name = "hack_motion"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team 049",
    description="Motion server: IK, joint moves with closed loop on /joint_states, hand presets.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["motion_node = hack_motion.motion_node:main"]},
)
