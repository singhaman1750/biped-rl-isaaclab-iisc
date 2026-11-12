#!/bin/bash
# Sandbox only. The two co-optimisation tasks again, after the train.py copy fix, once the first batch is done.
S=/tmp/symsandbox_v2_20261020
until grep -q DONE $S/smoke_status.txt; do sleep 10; done
cd $S
for spec in "sf_copt2 Isaac-Limx-SF-Copt-MoRAL-Rough-v0" "quad_copt2 Isaac-Quadruped-Copt-Rough-v0"; do
  set -- $spec
  name=$1; task=$2
  PYTHONPATH=$S/environments:$S/himloco timeout 2400 /workspace/isaaclab/isaaclab.sh -p $S/scripts/rsl_rl/train.py \
    --num_envs 64 --max_iterations 3 --headless --device cuda:0 --task $task --policy-type COPT \
    --design_search random --copt_num_individuals 4 > $S/smoke_$name.log 2>&1
  code=$?
  echo "$name exit $code tracebacks $(grep -ac Traceback $S/smoke_$name.log) iterations $(grep -ac 'Learning iteration' $S/smoke_$name.log)" >> $S/smoke_status.txt
done
echo DONE2 >> $S/smoke_status.txt
