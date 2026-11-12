#!/bin/bash
# Section 9.3 of plans/symmetry_tron1_quadruped.md, on the live tree after application.
W=/tmp/symmetry_smoke; mkdir -p $W; cd $W; : > status.txt
T=/workspace/isaaclab/biped/scripts/rsl_rl/train.py
A="--num_envs 64 --max_iterations 3 --headless --device cuda:0"
C="--policy-type COPT --design_search random --copt_num_individuals 4"
i=0
for t in "Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0" "Isaac-Quadruped-Blind-Rough-v0" \
         "Isaac-Limx-SF-HIM-Identified-Blind-Rough-Urdf-v0 --policy-type HIMPPO" \
         "Isaac-Quadruped-HIM-Blind-Rough-v0 --policy-type HIMPPO" \
         "Isaac-Limx-SF-Copt-MoRAL-Rough-v0 $C" "Isaac-Quadruped-Copt-Rough-v0 $C" \
         "Isaac-Limx-SDBRS1-Blind-Rough-v0" "Isaac-Limx-Kscale-Blind-Rough-v0"; do
  i=$((i+1))
  timeout 2400 /workspace/isaaclab/isaaclab.sh -p $T --task $t $A > run_$i.log 2>&1
  echo "$i $t | exit $? tracebacks $(grep -ac Traceback run_$i.log) | $(grep -a '\[symmetry\]' run_$i.log) | $(grep -a 'Mean symmetry loss\|  symmetry: ' run_$i.log | tr -s ' ' | tr '\n' ' ')" >> status.txt
done
echo DONE >> status.txt
