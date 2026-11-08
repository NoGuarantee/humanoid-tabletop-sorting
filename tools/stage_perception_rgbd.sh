#!/bin/bash
# Extract the verified private-release archive supplied by the operator.
set -eu
cd "$(dirname "$0")/.."
archive=weights/geometry_only_ros_bundle.tar.gz
digest=b33b2debe3c1710bd6e7458b983767bfc7a7ce003b0d9c5560ad20ec5bd7fade
if command -v sha256sum >/dev/null; then
    printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
else
    printf '%s  %s\n' "$digest" "$archive" | shasum -a 256 -c -
fi
stage=$(mktemp -d weights/.rgbd-stage.XXXXXX)
trap 'rm -rf "$stage"' EXIT
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
