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
# one path segment: a slash or ".." would point this dev container at another name's copy
case "$WHO" in
  ""|*[!A-Za-z0-9_-]*) echo "[bench] NSSR_WHO must be one name of letters, digits, _ and -" >&2; exit 1 ;;
esac
WORK=$BASE/work/$WHO/nssr-sol-team-049
LOGS=$BASE/logs
LOCK=$BASE/bench.lock
SIM=${SIM_CONTAINER:-nssr-sim}
SOL=${SOLUTION_CONTAINER:-nssr-solution}
IMAGE=${SIM_IMAGE:-nssr-sim:latest}
INSTALLED=nssr-solution-installed:$(printf '%s' "$WHO" | tr -c 'a-zA-Z0-9_.-' '_')
DEV_REPO=${DEV_IMAGE_REPO:-nssr-dev}
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
  # --no-daemon: a ros2 daemon spawned while the sim is still loading breaks (!rclpy.ok()) and then every
  # daemon-backed call (echo, topic list) fails until it is restarted.
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$1" ]; do
    has sim || { echo "[bench] the simulator exited:"; tail_log sim 40; return 1; }
    if ros_cmd 15 "ros2 topic echo --once --no-daemon --qos-durability transient_local --qos-reliability reliable \
/sim/task" >/dev/null 2>&1; then
      ros_cmd 30 "ros2 daemon stop; ros2 daemon start" >/dev/null 2>&1 || true   # fresh daemon for remote.sh ros
      echo "[bench] /sim/task published after $((SECONDS - t0)) s"; return 0
    fi
    sleep 5
  done
  echo "[bench] no /sim/task after $1 s:"; tail_log sim 30; return 1
}

sim_start() {  # task.sh arguments; --force takes the bench over; without a display the sim runs headless
  local force=0 envs="" args=() a
  for a in "$@"; do
    case "$a" in
      --force) force=1 ;;
      --self-collision) envs="HACK_SELF_COLLISION=1 " ;;  # debug only: off at the final check
      *) args+=("$a") ;;
    esac
  done
  [ ${#args[@]} -gt 0 ] || die "usage: sim start sort3 --pool 4 [--seed S] [--force] [--self-collision]"
  [ -n "${DISPLAY:-}" ] || [[ " ${args[*]} " == *" --headless "* ]] || args+=(--headless)
  take_bench "$force" "sim ${args[*]}"
  has solution && solution_stop
  has sim && { tmux kill-session -t sim; stop_sim_procs; }
  ( cd "$SIMREPO" && bash hackathon/docker/run.sh start >/dev/null ) || die "cannot start container $SIM"
  : > "$LOGS/sim.log"
  local q; q=$(printf '%q ' "${args[@]}")
  tmux new-session -d -s sim "cd '$SIMREPO' && ${envs}bash hackathon/tools/task.sh $q 2>&1 | tee -a '$LOGS/sim.log'"
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

solution_wait() {  # $1=timeout s: until /sim/task/done appears in the log and run.sh ends, or the solution exits
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$1" ]; do
    if grep -q "published /sim/task/done" "$LOGS/solution.log" 2>/dev/null; then
      echo "[bench] /sim/task/done after $((SECONDS - t0)) s (wall clock from the solution start; includes the install unless --layouts)"
      solution_wait_exit 60; return $?
    fi
    has solution || { echo "[bench] the solution exited without /sim/task/done"; return 1; }
    sleep 1  # "run.sh ended N s after /sim/task/done" counts from here: a longer sleep hides seconds of N
  done
  echo "[bench] timeout $1 s: no /sim/task/done"; return 1
}

solution_wait_exit() {  # $1=timeout s: the final check expects run.sh to end after /sim/task/done
  local t0=$SECONDS
  while has solution; do
    [ $((SECONDS - t0)) -lt "$1" ] || { echo "[bench] run.sh still running $1 s after /sim/task/done"; return 1; }
    sleep 1
  done
  echo "[bench] run.sh ended $((SECONDS - t0)) s after /sim/task/done"
}

solution_docker() {  # $1=image, $2=script; the rest are extra docker run options. Same container as solution.sh, no TTY
  local image=$1 script=$2; shift 2
  docker run --name "$SOL" --gpus all --network host --ipc host "$@" \
    -v "$DDS:/etc/nssr/fastdds.xml:ro" \
    -e ROS_DOMAIN_ID=146 -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/nssr/fastdds.xml \
    --entrypoint bash "$image" -c "$script"
}

solution_install() {  # the install of hackathon/tools/solution.sh, kept as an image: it runs once, run.sh on every layout
  need_bench
  [ -f "$WORK/run.sh" ] || die "no $WORK/run.sh: remote.sh sync"
  has solution && solution_stop
  docker rm -f "$SOL" >/dev/null 2>&1
  : > "$LOGS/solution.log"
  echo "[bench] installing the solution as in solution.sh"
  solution_docker "$IMAGE" '
set -e
mkdir -p /solution && cp -a /mnt/solution/. /solution/ && cd /solution && rm -rf build install log
if [ -s apt.txt ]; then apt-get update -qq && xargs -a apt.txt apt-get install -y --no-install-recommends; fi
if [ -s requirements.txt ]; then /usr/bin/python3 -m pip install -r requirements.txt; fi
if [ -f setup.sh ]; then bash setup.sh; fi
source /opt/ros/humble/setup.bash
colcon build' -v "$WORK:/mnt/solution:ro" 2>&1 | clean >> "$LOGS/solution.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { docker rm -f "$SOL" >/dev/null 2>&1; echo "[bench] install failed, exit $rc:"; tail_log solution 40; return 1; }
  docker commit "$SOL" "$INSTALLED" >/dev/null && docker rm "$SOL" >/dev/null
  cp "$LOGS/solution.log" "$LOGS/solution.install.log"  # the layouts start solution.log anew
  clean < "$LOGS/solution.log" | grep -E "^\[(curobo_install|curobo_warmup|setup)\]|^Summary:" | cut -c1-200
}

mem_peak() {  # what the budget of the final check is about, simulator and solution together, while run.sh runs
  local v r vmax=0 rmax=0 i=0
  while has solution; do
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    r=$(free -m | awk '/^Mem:/{print $3}')
    [ "$v" -gt "$vmax" ] && vmax=$v
    [ "$r" -gt "$rmax" ] && rmax=$r
    i=$((i + 1))
    [ "$i" = 15 ] && gpu | grep -E "MiB|Mem:|nssr-"  # a breakdown while cuRobo is up, if run.sh lasts that long
    sleep 1
  done
  echo "[bench] peak while run.sh ran: VRAM $vmax MiB, RAM $rmax MiB (budget 16 GB, 32 GB; sampled once a second)"
}

layout_start() {  # run.sh in a clean container from the installed image; a new log for each layout
  echo "[bench] layout start $(date +%s.%N)" > "$LOGS/solution.log"  # t=0 for tools/startup_report.py
  local script="export LAYOUT=$1; cd /solution && source install/setup.bash && ${RUN_CMD:-bash run.sh}"
  tmux new-session -d -s solution "docker run --rm --name '$SOL' --gpus all --network host --ipc host \
-v '$DDS:/etc/nssr/fastdds.xml:ro' -e ROS_DOMAIN_ID=146 -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/nssr/fastdds.xml \
--entrypoint bash '$INSTALLED' -c '$script' 2>&1 | tee -a '$LOGS/solution.log'"
}

eval_layouts() {  # $1=layouts, $2=timeout s for each. As at the final check: one simulator, run.sh anew on each layout
  local n=$1 timeout=$2 i seed mem rc=0
  solution_install || return 1
  for i in $(seq 1 "$n"); do
    if [ "$i" -gt 1 ]; then  # moves the objects only, the robot stays where the last run left it
      seed=$RANDOM
      ( cd "$SIMREPO" && bash hackathon/tools/objects.sh randomize "$seed" ) || { rc=1; break; }
      sleep 3
    fi
    echo "===== layout $i/$n, seed ${seed:-of the sim start}"
    layout_start "$i"
    sleep 1; { mem_peak | tee -a "$LOGS/solution.log"; } & mem=$!
    # into the log too: stdout dies with the ssh connection, the log stays on the server
    solution_wait "$timeout" | tee -a "$LOGS/solution.log"; [ "${PIPESTATUS[0]}" = 0 ] || rc=1
    has solution && solution_stop
    wait "$mem"
    echo "----- solution log (start-up and end)"
    clean < "$LOGS/solution.log" | grep -E "limit [0-9]+ s|curobo|startup:|escalating|published /sim/task/done|mission failed" | tail -n 12
    python3 "$WORK/tools/startup_report.py" "$LOGS/solution.log" 2>&1 | tail -n 25
    cp "$LOGS/solution.log" "$LOGS/solution.layout$i.log"
    echo "----- self-check"; ( cd "$SIMREPO" && bash hackathon/tools/task.sh check )
  done
  docker rmi "$INSTALLED" >/dev/null 2>&1
  return $rc
}

dev_tag() {  # hash of what the install reads in your synced copy: changing any of these needs a new image
  ( cd "$WORK" && cat setup.sh tools/curobo_install.sh tools/curobo_warmup.py requirements.txt apt.txt 2>/dev/null ) \
    | sha256sum | cut -c1-12
}

dev_image() {  # $1=--force rebuilds: the install of solution.sh, frozen into an image for `dev`. `eval` never uses it.
  [ -f "$WORK/setup.sh" ] || die "no $WORK/setup.sh: remote.sh sync"
  local img name rc
  img="$DEV_REPO:$(dev_tag)"; name="nssr-dev-build-$WHO"
  # Agents keep this tag. A new hash gets its own tag; this one is never moved.
  if [ "$img" = "nssr-dev:6057b5989a06" ] && docker image inspect "$img" >/dev/null 2>&1; then
    echo "[bench] $img is kept as-is"; return 0
  fi
  if [ "${1:-}" != --force ] && docker image inspect "$img" >/dev/null 2>&1; then
    echo "[bench] $img is already built"; return 0
  fi
  if docker image inspect "$img" >/dev/null 2>&1; then
    local users
    users=$(docker ps -a --filter "ancestor=$img" --format '{{.Names}}' | grep -v "^${name}$" || true)
    [ -z "$users" ] || die "refusing to overwrite $img; containers still use it: $users"
  fi
  docker rm -f "$name" >/dev/null 2>&1
  echo "[bench] building $img (log: remote.sh logs dev-image)"
  # setup.sh never fails on purpose (see its header), so the install is checked here before anything is frozen
  docker run --name "$name" --gpus all --network host -v "$WORK:/mnt/solution:ro" --entrypoint bash "$IMAGE" -c '
set -e
mkdir -p /tmp/dev-image && cp -a /mnt/solution/. /tmp/dev-image/ && cd /tmp/dev-image && rm -rf build install log
if [ -s apt.txt ]; then apt-get update -qq && xargs -a apt.txt apt-get install -y --no-install-recommends; fi
if [ -s requirements.txt ]; then /usr/bin/python3 -m pip install -r requirements.txt; fi
bash setup.sh
/usr/bin/python3 -c "import curobo, torch, warp; assert torch.cuda.is_available(), \"no CUDA in torch\""
test -n "$(ls -A /root/.cache/warp)"
cd / && rm -rf /tmp/dev-image /root/.cache/pip' 2>&1 | tee "$LOGS/dev-image.log" | grep -E 'curobo_install\]|CUROBO_WARMUP|FAILED|Error'
  rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    docker rm -f "$name" >/dev/null 2>&1
    die "the install failed (rc $rc), no image: remote.sh logs dev-image"
  fi
  docker commit --change 'ENTRYPOINT ["bash"]' --change 'CMD []' "$name" "$img" >/dev/null || die "docker commit failed"
  docker rm -f "$name" >/dev/null 2>&1
  echo "[bench] built $img: remote.sh dev now starts from it, drop 'bash setup.sh' from dev commands"
}

dev_name() { echo "nssr-dev-$WHO"; }

dev_stop_cmd() {  # $1=container. The previous command in our container only; pid 1 is sleep and stays.
  docker exec "$1" bash -c 'pid=$(cat /tmp/nssr-dev.pid 2>/dev/null || true)
[ -n "${pid:-}" ] && [ "$pid" != 1 ] || exit 0
pkill -TERM -P "$pid" 2>/dev/null || true
kill -TERM "$pid" 2>/dev/null || true
sleep 1
pkill -KILL -P "$pid" 2>/dev/null || true
kill -KILL "$pid" 2>/dev/null || true
rm -f /tmp/nssr-dev.pid' >/dev/null 2>&1 || true
}

dev_up() {  # $1=image to create from. Reuse only our keepalive container, and only when /solution is $WORK.
  local img=$1 name running mount cimg label
  name=$(dev_name)
  running=$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || echo no)
  mount=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/solution"}}{{.Source}}{{end}}{{end}}' "$name" 2>/dev/null || true)
  cimg=$(docker inspect -f '{{.Config.Image}}' "$name" 2>/dev/null || true)
  label=$(docker inspect -f '{{index .Config.Labels "nssr.dev"}}' "$name" 2>/dev/null || true)
  if [ "$running" = true ] && [ "$mount" = "$WORK" ] && [ "$label" = 1 ]; then
    if [ "$cimg" = "$img" ]; then
      echo "[bench] reusing $name" >&2
      dev_stop_cmd "$name"
      return 0
    fi
    # This tree's tag is not built. Do not throw away a container that already has torch and cuRobo.
    if [ "$img" = "$IMAGE" ] && [[ "$cimg" == "$DEV_REPO:"* ]]; then
      echo "[bench] reusing $name on $cimg; the tag for this tree is not built" >&2
      dev_stop_cmd "$name"
      return 0
    fi
  fi
  if [ "$running" != no ]; then
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi
  echo "[bench] starting $name from $img" >&2
  docker run -d --name "$name" --label nssr.dev=1 --gpus all --network host --ipc host \
    -v "$WORK:/solution" -v "$DDS:/etc/nssr/fastdds.xml:ro" \
    -e ROS_DOMAIN_ID=146 -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/nssr/fastdds.xml \
    -w /solution --entrypoint bash "$img" -c 'sleep infinity' >/dev/null \
    || die "cannot start $name"
}

dev_exec() {  # $1=timeout, rest = command. install/ is sourced only when it is not older than the synced sources.
  local t=$1; shift
  docker exec -e CMD="$*" -e T="$t" "$(dev_name)" bash -c 'source /opt/ros/humble/setup.bash
if [ -f install/setup.bash ] && [ -z "$(find src -type f -newer install/setup.bash \
  ! -path "*/__pycache__/*" ! -path "*.egg-info/*" -print -quit 2>/dev/null)" ]; then
  source install/setup.bash
elif [ -f install/setup.bash ]; then
  echo "[bench] install/ is older than the synced sources, not sourcing it" >&2
fi
echo $$ > /tmp/nssr-dev.pid
timeout "$T" bash -c "$CMD"
rc=$?
rm -f /tmp/nssr-dev.pid
exit $rc'
}

status() {
  echo "bench:$( [ -f "$LOCK" ] && cat "$LOCK" || echo free)"
  echo "you:      $WHO ($WORK $( [ -d "$WORK" ] && echo present || echo 'missing: remote.sh sync'))"
  echo "sessions: $(tmux ls -F '#S' 2>/dev/null | tr '\n' ' ')"
  echo "sim:      container $(sim_up && echo running || echo stopped), simulator $(has sim && echo running || echo stopped)"
  echo "solution: $(has solution && echo running || echo stopped)"
  if docker inspect "$(dev_name)" >/dev/null 2>&1; then
    echo "dev:      $(docker inspect -f '{{.State.Status}} {{.Config.Image}} mount {{range .Mounts}}{{if eq .Destination "/solution"}}{{.Source}}{{end}}{{end}}' "$(dev_name)")"
  else
    echo "dev:      stopped"
  fi
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
  sim start TASK [task.sh args]   e.g. sim start sort3 --pool 4 --seed 7; waits for /sim/task. --self-collision turns on physics self-collisions (debug). --force takes the bench
  sim stop | sim restart ...      stop the simulator and release the bench
  solution [start|stop|wait S]    solution.sh as at the final check, from your synced copy
  eval TASK [args] [--keep] [--timeout S] [--layouts N] [--run CMD]  sim start, solution, wait for done and exit, self-check, stop. --layouts: install once, then run.sh in a clean container on each of N layouts of one simulator; --timeout is per layout, --run replaces run.sh there (tools/startup_probe.py)
  check                           self-check (/sim/task/check, debug only: tools/, never src/)
  ros CMD...                      a ROS 2 command in the sim container, 20 s timeout (NSSR_ROS_TIMEOUT)
  dev [--timeout S] -- CMD        CMD in nssr-dev-<you>, your copy mounted at /solution. The container stays up.
                                  colcon build/ and install/ stay in that copy. A second dev with the same name
                                  stops the previous command in that container only. Uses the dev image for your
                                  setup.sh if it is built, otherwise an existing nssr-dev container for this name,
                                  otherwise the bare image
  dev-image [--force]             build that image: torch, cuRobo and the kernel cache, ~6 min once. Does not
                                  overwrite a tag a container is using, and does not replace nssr-dev:6057b5989a06
  logs sim|solution|setup|dev-image [N]  the last N lines (default 60)
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
    keep=0; timeout=900; layouts=0; args=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --keep) keep=1; shift ;;
        --timeout) timeout=$2; shift 2 ;;
        --layouts) layouts=$2; shift 2 ;;
        --run) RUN_CMD=$2; shift 2 ;;  # debug: instead of bash run.sh on each layout, no single quotes in it
        *) args+=("$1"); shift ;;
      esac
    done
    sim_start "${args[@]}" || exit 1
    if [ "$layouts" -gt 0 ]; then
      eval_layouts "$layouts" "$timeout"; rc=$?
    else
      solution_start
      solution_wait "$timeout"; rc=$?
      echo "----- solution log (tail)"; tail_log solution 40
      echo "----- self-check"; ( cd "$SIMREPO" && bash hackathon/tools/task.sh check )
    fi
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
    img="$DEV_REPO:$(dev_tag)"
    if ! docker image inspect "$img" >/dev/null 2>&1; then
      echo "[bench] no $img; reusing this name's nssr-dev container if it is already up, else $IMAGE: remote.sh dev-image" >&2
      img=$IMAGE
    fi
    dev_up "$img"
    dev_exec "$t" "$@" ;;
  dev-image) dev_image "$@" ;;
  logs) tail_log "${1:-sim}" "${2:-60}" ;;
  gpu) gpu ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac
