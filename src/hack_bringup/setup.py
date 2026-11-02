from glob import glob

from setuptools import setup

package_name = "hack_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team 049",
    description="Launch files and parameters of the team-049 solution.",
    license="Apache-2.0",
    entry_points={"console_scripts": []},
)
