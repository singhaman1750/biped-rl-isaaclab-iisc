#!/bin/bash
# Usage: bash cmaes_smoke.sh <task> <tag> [extra args]
set -u
TASK=$1; TAG=$2; shift 2
mkdir -p /tmp/cmaes_smoke/$TAG && cd /tmp/cmaes_smoke/$TAG
export PYTHONPATH=/workspace/isaaclab/biped/co_optimisation:${PYTHONPATH:-}
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/biped/scripts/rsl_rl/train.py \
  --task "$TASK" --headless --policy-type COPT --num_envs 1024 --copt_num_individuals 16 \
  --ea_late_start 4 --ea_update_interval 2 --max_iterations 14 --save_interval 5 "$@" \
  > /tmp/cmaes_smoke/$TAG/run.log 2>&1
echo "EXIT $?" >> /tmp/cmaes_smoke/$TAG/run.log
