#!/bin/bash
# Prepares a fresh GPU server (vast.ai VM, Ubuntu 22.04+) for the bench: Docker with GPU, tmux, gh, the simulator repo,
# its assets, the image nssr-sim:latest and the container. Safe to run again: finished steps are skipped.
# From your machine: bash tools/remote.sh setup      (or on the server: bash server_setup.sh)
# Before the first run, log in to GitHub on the server with a READ-ONLY token, see docs/SERVER.md.
set -eu
BASE=${NSSR_BASE:-$HOME/nssr}
SIMREPO=${NSSR_SIM:-$BASE/nssr-uwt-polygon}
mkdir -p "$BASE/work" "$BASE/logs"
exec > >(tee -a "$BASE/logs/setup.log") 2>&1
echo "===== setup $(date '+%F %T')"
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO=sudo
step() { echo; echo "[setup] $*"; }
apt_install() { $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "$@"; }

step "GPU and driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || { echo "no nvidia-smi: this machine has no working NVIDIA driver"; exit 1; }
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 580 ] || echo "[WARN] driver $drv < 580: the organisers require 580+"
# unattended-upgrades updates the NVIDIA user-space libraries under a running kernel module, and then every
# `docker run --gpus` fails with "Driver/library version mismatch" until a reboot (happened 2026-11-03).
# Keep the driver fixed (apt-mark hold below, after the Container Toolkit) and the security updates manual.
if [ -f /etc/apt/apt.conf.d/20auto-upgrades ]; then
  $SUDO sed -i 's/^APT::Periodic::Unattended-Upgrade "1";/APT::Periodic::Unattended-Upgrade "0";/' \
    /etc/apt/apt.conf.d/20auto-upgrades
fi

step "tools: tmux, rsync, git, curl"
command -v tmux >/dev/null && command -v rsync >/dev/null && command -v git >/dev/null && command -v curl >/dev/null \
  || apt_install tmux rsync git curl ca-certificates gnupg

step "Docker"
if ! command -v docker >/dev/null; then
  # official apt repository: https://docs.docker.com/engine/install/ubuntu/
  $SUDO install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi
if ! docker info >/dev/null 2>&1; then
  $SUDO usermod -aG docker "$USER"
  echo "added $USER to the docker group: log out, log in and run setup again"; exit 1
fi

step "NVIDIA Container Toolkit"
if ! docker run --rm --gpus all ubuntu nvidia-smi -L >/dev/null 2>&1; then
  # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  apt_install nvidia-container-toolkit
  $SUDO nvidia-ctk runtime configure --runtime=docker
  $SUDO systemctl restart docker
  docker run --rm --gpus all ubuntu nvidia-smi -L || { echo "containers still do not see the GPU"; exit 1; }
fi
docker run --rm --gpus all ubuntu nvidia-smi -L

step "hold the NVIDIA driver and Container Toolkit packages"
# status "hi" means already held; a second setup must not call apt-mark with an empty list
pkgs=$(dpkg -l | awk '/^[hi]i +(nvidia-|libnvidia-)/ {print $2}')
# shellcheck disable=SC2086
[ -n "$pkgs" ] && $SUDO apt-mark hold $pkgs >/dev/null
apt-mark showhold | grep -c -E '^(nvidia-|libnvidia-)' | sed 's/^/held packages: /'

step "GitHub CLI"
if ! command -v gh >/dev/null; then
  # the Ubuntu 22.04 package is too old: https://github.com/cli/cli/blob/trunk/docs/install_linux.md
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | $SUDO dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg status=none
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
https://cli.github.com/packages stable main" | $SUDO tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  apt_install gh
fi
gh_ok=0
gh auth status >/dev/null 2>&1 && gh_ok=1
usd_size=$(stat -c %s "$SIMREPO/assets/robots/tiangong_pro/tiangong_pro_v2.usd" 2>/dev/null || echo 0)
if [ "$gh_ok" -ne 1 ]; then
  if [ -d "$SIMREPO/.git" ] && [ "$usd_size" -ge 1000000 ]; then
    echo "[setup] no GitHub login; using the simulator copy already at $SIMREPO"
  else
    echo "not logged in to GitHub. On the server, with a read-only token (docs/SERVER.md):"
    echo "  gh auth login --with-token <<< 'github_pat_...' && gh auth setup-git"
    echo "or copy your clone to $SIMREPO (assets included) and run setup again"
    exit 1
  fi
fi

step "simulator repo: $SIMREPO"
if [ -d "$SIMREPO/.git" ]; then
  if [ "$gh_ok" -eq 1 ]; then git -C "$SIMREPO" pull --ff-only
  else echo "[setup] skip git pull: not logged in to GitHub"; fi
else gh repo clone nssr-hackathon/nssr-uwt-polygon "$SIMREPO"; fi
cd "$SIMREPO"

step "assets"
if [ "$(stat -c %s assets/robots/tiangong_pro/tiangong_pro_v2.usd 2>/dev/null || echo 0)" -lt 1000000 ]; then
  bash hackathon/tools/get_assets.sh
fi

step "image nssr-sim:latest (the first build downloads ~8.5 GB and takes ~10 min)"
docker image inspect nssr-sim:latest >/dev/null 2>&1 || bash hackathon/docker/run.sh build

step "simulator container"
bash hackathon/docker/run.sh start
bash hackathon/docker/run.sh init
bash hackathon/docker/run.sh check

step "done. From your machine: bash tools/remote.sh eval sort3 --pool 4"
