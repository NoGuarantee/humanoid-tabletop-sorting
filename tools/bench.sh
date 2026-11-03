#!/bin/bash
# Server side of tools/remote.sh: the simulator and the solution on the shared GPU server ("the bench").
# Usually called through tools/remote.sh, which pipes this file over ssh, so the server always runs the version from
# your checkout. Can be run on the server directly too: NSSR_WHO=name bash tools/bench.sh status
#
# Layout on the server (override in ~/nssr/bench.env):
#   ~/nssr/nssr-uwt-polygon            simulator repo, image nssr-sim:latest, container nssr-sim (tools/server_setup.sh)
#   ~/nssr/work/<who>/nssr-sol-team-049 your copy of this repo (remote.sh sync)
#   ~/nssr/logs/                       sim.log, solution.log, ros.log; the simulator also writes its own logs/ in its repo
#   ~/nssr/bench.lock                  who holds the bench: one simulator per GPU, one ROS domain, one solution container
set -u
BASE=${NSSR_BASE:-$HOME/nssr}
[ -f "$BASE/bench.env" ] && . "$BASE/bench.env"
SIMREPO=${NSSR_SIM:-$BASE/nssr-uwt-polygon}
WHO=${NSSR_WHO:-$USER}
WORK=$BASE/work/$WHO/nssr-sol-team-049
LOGS=$BASE/logs
LOCK=$BASE/bench.lock
SIM=${SIM_CONTAINER:-nssr-sim}
SOL=${SOLUTION_CONTAINER:-nssr-solution}
IMAGE=${SIM_IMAGE:-nssr-sim:latest}
DDS=$SIMREPO/hackathon/docker/fastdds_hack.xml
ROS_PRE="source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=146; \
export FASTRTPS_DEFAULT_PROFILES_FILE=/nssr_sim/hackathon/docker/fastdds_hack.xml"
mkdir -p "$LOGS"

die() { echo "[bench] $*" >&2; exit 1; }
has() { tmux has-session -t "$1" 2>/dev/null; }
sim_up() { docker ps --format '{{.Names}}' | grep -qx "$SIM"; }
clean() { tr -d '\r' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g'; }   # docker -t output: CR and colour codes
tail_log() { [ -f "$LOGS/$1.log" ] || die "no log $LOGS/$1.log"; tail -n "${2:-60}" "$LOGS/$1.log" | clean; }
owner() { [ -f "$LOCK" ] && cut -d' ' -f1 "$LOCK"; }

need_bench() {  # the caller must hold the bench
  local o; o=$(owner)
  [ "$o" = "$WHO" ] || die "the bench is held by '${o:-nobody}', not '$WHO': remote.sh sim start ..."
}

take_bench() {  # $1=force, rest = description
  local force=$1; shift
  local o; o=$(owner)
  if [ -n "$o" ] && [ "$o" != "$WHO" ] && has sim && [ "$force" != 1 ]; then
    die "the bench is held: $(cat "$LOCK"). Ask them, or take it over with --force"
  fi
  echo "$WHO $(date '+%F %T') $*" > "$LOCK"
}

stop_sim_procs() {  # same as run_sim.sh: stop the sim and its bridges inside the container
  sim_up || return 0
  docker exec "$SIM" bash -c "pkill -f '[h]ack_sim_timer'; pkill -INT -f '[s]im_runner.py'; sleep 8; \
pkill -f '[r]os2_zmq_bridge'; pkill -f '[c]amera_bridge.py'; pkill -f '[r]obot_state_publisher'; pkill -f '[s]im_node.py'; \
pkill -9 -f '[s]im_runner.py'; true"
}

ros_cmd() {  # $1=timeout, rest = command run in the sim container with ROS 2 sourced (like ros.sh, without a TTY)
  local t=$1; shift
  sim_up || die "container $SIM is not running: remote.sh sim start ..."
  docker exec -e CMD="$*" "$SIM" bash -c "$ROS_PRE; timeout $t bash -c \"\$CMD\""
}

wait_task() {  # $1=timeout s: wait until the simulator publishes /sim/task (latched)
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$1" ]; do
    has sim || { echo "[bench] the simulator exited:"; tail_log sim 40; return 1; }
    if ros_cmd 15 "ros2 topic echo --once --qos-durability transient_local --qos-reliability reliable /sim/task" \
        >/dev/null 2>&1; then
      echo "[bench] /sim/task published after $((SECONDS - t0)) s"; return 0
    fi
    sleep 5
  done
  echo "[bench] no /sim/task after $1 s:"; tail_log sim 30; return 1
}

sim_start() {  # task.sh arguments; --force takes the bench over; without a display the sim runs headless
  local force=0 args=() a
  for a in "$@"; do [ "$a" = --force ] && force=1 || args+=("$a"); done
  [ ${#args[@]} -gt 0 ] || die "usage: sim start sort3 --pool 4 [--seed S] [--force]"
  [ -n "${DISPLAY:-}" ] || [[ " ${args[*]} " == *" --headless "* ]] || args+=(--headless)
  take_bench "$force" "sim ${args[*]}"
  has solution && solution_stop
  has sim && { tmux kill-session -t sim; stop_sim_procs; }
  ( cd "$SIMREPO" && bash hackathon/docker/run.sh start >/dev/null ) || die "cannot start container $SIM"
  : > "$LOGS/sim.log"
  local q; q=$(printf '%q ' "${args[@]}")
  tmux new-session -d -s sim "cd '$SIMREPO' && bash hackathon/tools/task.sh $q 2>&1 | tee -a '$LOGS/sim.log'"
  echo "[bench] sim started: ${args[*]} (log: remote.sh logs sim)"
  wait_task "${NSSR_SIM_WAIT:-300}"
}

sim_stop() {
  local o; o=$(owner)
  if [ -n "$o" ] && [ "$o" != "$WHO" ] && [ "${1:-}" != --force ]; then
    die "the bench is held: $(cat "$LOCK"). Ask them, or stop with --force"
  fi
  has solution && solution_stop
  stop_sim_procs
  has sim && tmux kill-session -t sim
  rm -f "$LOCK"
  echo "[bench] sim stopped, bench released"
}

solution_start() {
  need_bench
  has sim || die "no simulator running: remote.sh sim start ..."
  [ -f "$WORK/run.sh" ] || die "no $WORK/run.sh: remote.sh sync"
  has solution && solution_stop
  : > "$LOGS/solution.log"
  tmux new-session -d -s solution "bash '$SIMREPO/hackathon/tools/solution.sh' '$WORK' 2>&1 | tee -a '$LOGS/solution.log'"
  echo "[bench] solution started from $WORK (log: remote.sh logs solution)"
}

solution_stop() {
  docker rm -f "$SOL" >/dev/null 2>&1
  has solution && tmux kill-session -t solution
  echo "[bench] solution stopped"
}

solution_wait() {  # $1=timeout s: until /sim/task/done appears in the log or the solution exits
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$1" ]; do
    if grep -q "published /sim/task/done" "$LOGS/solution.log" 2>/dev/null; then
      echo "[bench] /sim/task/done after $((SECONDS - t0)) s (wall clock, including install)"; return 0
    fi
    has solution || { echo "[bench] the solution exited without /sim/task/done"; return 1; }
    sleep 5
  done
  echo "[bench] timeout $1 s: no /sim/task/done"; return 1
}

status() {
  echo "bench:    $( [ -f "$LOCK" ] && cat "$LOCK" || echo free)"
  echo "you:      $WHO ($WORK $( [ -d "$WORK" ] && echo present || echo 'missing: remote.sh sync'))"
  echo "sessions: $(tmux ls -F '#S' 2>/dev/null | tr '\n' ' ')"
  echo "sim:      container $(sim_up && echo running || echo stopped), simulator $(has sim && echo running || echo stopped)"
  echo "solution: $(has solution && echo running || echo stopped)"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/gpu:      /'
}

gpu() {  # memory against the final check budget: RTX 5080 16 GB VRAM, 32 GB RAM, simulator and solution together
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
  echo; free -h | head -2
  echo; docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' 2>/dev/null
  echo; echo "budget at the final check: 16 GB VRAM, 32 GB RAM (simulator + solution)"
}

usage() {
  cat <<'TXT'
bench.sh (usually through tools/remote.sh):
  status                          who holds the bench, what is running, GPU
  sim start TASK [task.sh args]   e.g. sim start sort3 --pool 4 --seed 7; waits for /sim/task. --force takes the bench
  sim stop | sim restart ...      stop the simulator and release the bench
  solution [start|stop|wait S]    solution.sh as at the final check, from your synced copy
  eval TASK [args] [--keep] [--timeout S]  sim start, solution, wait for done, self-check, stop
  check                           self-check (/sim/task/check, debug only: tools/, never src/)
  ros CMD...                      a ROS 2 command in the sim container, 20 s timeout (NSSR_ROS_TIMEOUT)
  dev [--timeout S] -- CMD        CMD in a fresh container from the image, your copy at /solution (colcon, ros2 run)
  logs sim|solution|setup [N]     the last N lines (default 60)
  gpu                             VRAM, RAM, containers
TXT
}

cmd=${1:-help}; shift || true
case "$cmd" in
  status) status ;;
  sim)
    sub=${1:-}; shift || true
    case "$sub" in
      start) sim_start "$@" ;;
      stop) sim_stop "$@" ;;
      restart) sim_stop && sim_start "$@" ;;
      *) die "usage: sim start|stop|restart" ;;
    esac ;;
  solution)
    sub=${1:-start}; shift || true
    case "$sub" in
      start) solution_start ;;
      stop) need_bench; solution_stop ;;
      wait) solution_wait "${1:-600}"; tail_log solution 40 ;;
      *) die "usage: solution start|stop|wait S" ;;
    esac ;;
  eval)
    keep=0; timeout=900; args=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --keep) keep=1; shift ;;
        --timeout) timeout=$2; shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    sim_start "${args[@]}" || exit 1
    solution_start
    solution_wait "$timeout"; rc=$?
    echo "----- solution log (tail)"; tail_log solution 40
    echo "----- self-check"; ( cd "$SIMREPO" && bash hackathon/tools/task.sh check )
    seed=$(grep -o 'layout seed [0-9]*' "$LOGS/sim.log" | head -1)
    echo "----- ${seed:-layout seed unknown}"
    if [ "$keep" = 1 ]; then echo "[bench] --keep: sim and solution left running"; else sim_stop; fi
    exit $rc ;;
  check) ( cd "$SIMREPO" && bash hackathon/tools/task.sh check ) ;;
  ros) ros_cmd "${NSSR_ROS_TIMEOUT:-20}" "$@" ;;
  dev)
    t=600
    while [ $# -gt 0 ] && [ "$1" != -- ]; do
      case "$1" in --timeout) t=$2; shift 2 ;; *) die "usage: dev [--timeout S] -- CMD" ;; esac
    done
    shift || true
    [ $# -gt 0 ] || die "usage: dev [--timeout S] -- CMD"
    [ -d "$WORK" ] || die "no $WORK: remote.sh sync"
    name="nssr-dev-$WHO"
    docker rm -f "$name" >/dev/null 2>&1
    docker run --rm --name "$name" --gpus all --network host --ipc host \
      -v "$WORK:/solution" -v "$DDS:/etc/nssr/fastdds.xml:ro" \
      -e ROS_DOMAIN_ID=146 -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/nssr/fastdds.xml -e CMD="$*" -e T="$t" \
      -w /solution --entrypoint bash "$IMAGE" -c 'source /opt/ros/humble/setup.bash; \
        [ -f install/setup.bash ] && source install/setup.bash; timeout "$T" bash -c "$CMD"' ;;
  logs) tail_log "${1:-sim}" "${2:-60}" ;;
  gpu) gpu ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac
