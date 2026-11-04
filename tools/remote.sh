#!/bin/bash
# The GPU server from your machine: sync your working tree, run the simulator and the solution, read logs.
# Code, git and your keys stay here; the server only runs things. Server side: tools/bench.sh, piped over ssh on
# every call, so it always matches your checkout. New server: tools/server_setup.sh (see docs/SERVER.md).
#
# Settings: environment or .remote.env in the repo root (not in git):
#   NSSR_HOST=nssr      ssh host alias from ~/.ssh/config
#   NSSR_WHO=name       your folder on the server and your name in the bench lock (default: git user.name)
#
#   bash tools/remote.sh sync                      copy the working tree (no commit needed; .gitignore respected)
#   bash tools/remote.sh eval sort3 --pool 4       sync, simulator, solution as at the final check, self-check, stop
#   bash tools/remote.sh status | gpu | logs solution 100 | ros ros2 topic list
#   bash tools/remote.sh fetch logs/solution.log   copy a file or folder from ~/nssr on the server to .fetch/
#   bash tools/remote.sh help                      all bench commands
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$ROOT/.remote.env" ] && . "$ROOT/.remote.env"
HOST=${NSSR_HOST:-nssr}
WHO=${NSSR_WHO:-$(git -C "$ROOT" config user.name 2>/dev/null | tr -cs 'A-Za-z0-9_-' '_' | sed 's/_*$//')}
WHO=${WHO:-$USER}
DEST=nssr/work/$WHO/nssr-sol-team-049

bench() {  # run tools/bench.sh on the server with these arguments
  ssh "$HOST" "NSSR_WHO=$(printf '%q' "$WHO") bash -s -- $(printf '%q ' "$@")" < "$ROOT/tools/bench.sh"
}

sync() {
  ssh "$HOST" "mkdir -p $DEST"
  rsync -az --delete --exclude .git --exclude .fetch --filter=':- .gitignore' "$ROOT/" "$HOST:$DEST/"
  echo "[remote] synced to $HOST:~/$DEST"
}

cmd=${1:-help}
case "$cmd" in
  sync) sync ;;
  eval|solution)
    # the solution always runs from what you have now
    [ "$cmd" = solution ] && [ "${2:-start}" != start ] || sync
    bench "$@" ;;
  dev|dev-image) sync; bench "$@" ;;
  fetch)
    [ $# -ge 2 ] || { echo "usage: remote.sh fetch PATH_UNDER_~/nssr [LOCAL_DIR]"; exit 1; }
    out=${3:-$ROOT/.fetch}; mkdir -p "$out"
    rsync -az "$HOST:nssr/$2" "$out/"
    echo "[remote] $out/$(basename "$2")" ;;
  setup)
    # server_setup.sh is long (image build); its own log is ~/nssr/logs/setup.log
    ssh "$HOST" "bash -s" < "$ROOT/tools/server_setup.sh" ;;
  help|-h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; echo; bench help ;;
  *) bench "$@" ;;
esac
