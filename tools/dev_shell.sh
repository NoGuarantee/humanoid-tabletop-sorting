#!/bin/bash
# Interactive shell in a container from the simulator image with this repo mounted read-write at /solution,
# connected to the running simulator over ROS 2. Build inside with:
#   colcon build --symlink-install --build-base /tmp/b --install-base /tmp/i && source /tmp/i/setup.bash
# Set NSSR_SIM to the simulator repo (default: ../nssr-uwt-polygon next to this repo).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SIM=${NSSR_SIM:-$ROOT/../nssr-uwt-polygon}
DDS="$SIM/hackathon/docker/fastdds_hack.xml"
if [ ! -f "$DDS" ]; then
  echo "simulator repo not found: set NSSR_SIM=/path/to/nssr-uwt-polygon"
  exit 1
fi
IMAGE=${SIM_IMAGE:-nssr-sim:latest}
docker run --rm -it --name "nssr-dev-$USER" --gpus all --network host --ipc host \
  -v "$ROOT:/solution" -v "$DDS:/etc/nssr/fastdds.xml:ro" \
  -e ROS_DOMAIN_ID=146 -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/nssr/fastdds.xml \
  -w /solution --entrypoint bash "$IMAGE" -c 'source /opt/ros/humble/setup.bash && exec bash'
