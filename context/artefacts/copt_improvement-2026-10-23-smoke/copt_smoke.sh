#!/bin/bash
# Usage: bash copt_smoke.sh <task> <policy_type> <tag> [extra args]
set -u
TASK=$1; PT=$2; TAG=$3; shift 3
mkdir -p /tmp/copt_smoke/$TAG && cd /tmp/copt_smoke/$TAG
export PYTHONPATH=/workspace/isaaclab/biped/co_optimisation:${PYTHONPATH:-}
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/biped/scripts/rsl_rl/train.py \
  --task "$TASK" --headless --policy-type "$PT" --num_envs 1024 --copt_num_individuals 16 \
  --ea_late_start 4 --ea_update_interval 2 --max_iterations 8 --save_interval 5 "$@" \
  > /tmp/copt_smoke/$TAG/run.log 2>&1
echo "EXIT $?" >> /tmp/copt_smoke/$TAG/run.log
