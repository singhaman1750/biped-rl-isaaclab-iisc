#!/bin/bash
# Sandbox only. The live tree, unmodified, run from a directory holding no environments package, to attribute
# the SD_BRS1 HIM failure under its own symmetry configuration and the Go1 asset construction failure.
S=/tmp/symsandbox_v2_20261020
W=/tmp/symlive_20261020
mkdir -p $W
until grep -q DONE2 $S/smoke_status.txt; do sleep 10; done
cd $W
for spec in "live_brs_him2 Isaac-Limx-SDBRS1-HIM-Blind-Rough-v0 --policy-type HIMPPO" "live_go1 Isaac-Quadruped-Go1Asset-Blind-Rough-v0"; do
  set -- $spec
  name=$1; task=$2; shift 2
  timeout 1500 /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/biped/scripts/rsl_rl/train.py \
    --num_envs 64 --max_iterations 3 --headless --device cuda:0 --task $task "$@" > $S/smoke_$name.log 2>&1
  echo "$name exit $? tracebacks $(grep -ac Traceback $S/smoke_$name.log) iterations $(grep -ac 'Learning iteration' $S/smoke_$name.log)" >> $S/smoke_status.txt
done
echo DONE3 >> $S/smoke_status.txt
