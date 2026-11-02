#!/bin/bash
# Fails if the solution code (src/, run.sh, setup.sh) touches what the rules forbid at the final check:
# debug topics (calls to them are logged) and simulator internals (ground truth poses -> the task is annulled).
# Run before every push to main. Debug helpers belong in tools/, never in src/.
cd "$(dirname "$0")/.." || exit 1
fail=0
check() {  # pattern, reason
  hits=$(grep -rnE "$1" src run.sh setup.sh 2>/dev/null | grep -v '^src/hack_brain/hack_brain/brain_node.py:.*Never use the debug topics')
  if [ -n "$hits" ]; then echo "FORBIDDEN ($2):"; echo "$hits"; fail=1; fi
}
check '/sim/task/check|/sim/objects/set|/sim/objects/randomize' 'debug topic, off and logged at the final check'
check 'omni\.|isaacsim|isaaclab|pxr|\.usd\b|nssr_sim' 'simulator internals'
check 'ROS_DOMAIN_ID=|FASTRTPS_DEFAULT_PROFILES_FILE=' 'DDS settings are fixed by the evaluator container'
[ $fail -eq 0 ] && echo "check_rules: OK"
exit $fail
