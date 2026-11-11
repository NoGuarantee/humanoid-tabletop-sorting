#!/bin/bash
# Prefer the pinned archive shipped with PR9; an optional download supports source-only exports.
set -eu
cd "$(dirname "$0")/.."
archive=weights/geometry_only_ros_bundle.tar.gz
mkdir -p weights
if [ ! -f "$archive" ] && [ -f assets/perception/geometry_only_ros_bundle.tar.gz ]; then
    archive=assets/perception/geometry_only_ros_bundle.tar.gz
elif [ ! -f "$archive" ]; then
    url=${NSSR_RGBD_BUNDLE_URL:-https://github.com/nssr-hackathon/nssr-sol-team-049/releases/download/pr2-rgbd-model-20261105/geometry_only_ros_bundle.tar.gz}
    download=$(mktemp weights/.rgbd-download.XXXXXX)
    trap 'rm -f "$download"' EXIT
    curl --fail --location --silent --show-error --connect-timeout 10 --max-time 120 --retry 1 "$url" -o "$download"
    archive=$download
fi
digest=b33b2debe3c1710bd6e7458b983767bfc7a7ce003b0d9c5560ad20ec5bd7fade
if command -v sha256sum >/dev/null; then
    printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
else
    printf '%s  %s\n' "$digest" "$archive" | shasum -a 256 -c -
fi
stage=$(mktemp -d weights/.rgbd-stage.XXXXXX)
trap 'rm -rf "$stage"; if [ -n "${download:-}" ]; then rm -f "$download"; fi' EXIT
tar -xzf "$archive" -C "$stage"
# Publish an immutable checksum-addressed directory in one rename.
target="weights/rgbd-${digest:0:12}"
python3 - "$stage/bundle_box" "$target" <<'PYTHON'
import errno
import os
import sys

try:
    os.rename(sys.argv[1], sys.argv[2])
except OSError as exc:
    if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
        raise
PYTHON
test -f "$target/bundle.json"
