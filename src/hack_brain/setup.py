from glob import glob

from setuptools import setup

package_name = "hack_brain"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team 049",
    description="Task orchestrator: reads /sim/task, runs the pick-and-place state machine, publishes /sim/task/done.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["brain_node = hack_brain.brain_node:main"]},
)
