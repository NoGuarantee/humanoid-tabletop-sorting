from setuptools import find_packages, setup

package_name = "hack_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team 049",
    description="Perception service: RGB-D cameras to objects, trays and racks in the pelvis frame.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["perception_node = hack_perception.perception_node:main"]},
)
