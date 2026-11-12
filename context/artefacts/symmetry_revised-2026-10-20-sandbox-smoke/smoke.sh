#!/bin/bash
# Sandbox only. Three iterations of each representative task on 64 environments, from the sandbox tree.
S=/tmp/symsandbox_v2_20261020
cd $S
run() {
  name=$1; shift
  PYTHONPATH=$S/environments:$S/himloco timeout 1500 /workspace/isaaclab/isaaclab.sh -p $S/scripts/rsl_rl/train.py \
    --num_envs 64 --max_iterations 3 --headless --device cuda:0 "$@" > $S/smoke_$name.log 2>&1
  code=$?
  tb=$(grep -ac "Traceback" $S/smoke_$name.log)
  it=$(grep -ac "Learning iteration" $S/smoke_$name.log)
  echo "$name exit $code tracebacks $tb iterations $it" >> $S/smoke_status.txt
}
: > $S/smoke_status.txt
run sf_ppo   --task Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0
run quad_ppo --task Isaac-Quadruped-Blind-Rough-v0
run sf_him   --task Isaac-Limx-SF-HIM-Identified-Blind-Rough-Urdf-v0 --policy-type HIMPPO
run quad_him --task Isaac-Quadruped-HIM-Blind-Rough-v0 --policy-type HIMPPO
run sf_copt  --task Isaac-Limx-SF-Copt-MoRAL-Rough-v0 --policy-type COPT --design_search random --copt_num_individuals 4
run quad_copt --task Isaac-Quadruped-Copt-Rough-v0 --policy-type COPT --design_search random --copt_num_individuals 4
run go1_ppo  --task Isaac-Quadruped-Go1Asset-Blind-Rough-v0
run brs_ppo  --task Isaac-Limx-SDBRS1-Blind-Rough-v0
run brs_him  --task Isaac-Limx-SDBRS1-HIM-Blind-Rough-v0 --policy-type HIMPPO
run ks_ppo   --task Isaac-Limx-Kscale-Blind-Rough-v0
run ks_him   --task Isaac-Limx-Kscale-HIM-Blind-Rough-v0 --policy-type HIMPPO
run ks_yaw   --task Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-v0
# the live tree, unmodified, for the standing defect of the SD_BRS1 HIM task under its own symmetry configuration
legacy() {
  name=$1; shift
  timeout 1500 /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/biped/scripts/rsl_rl/train.py     --num_envs 64 --max_iterations 3 --headless --device cuda:0 "$@" > $S/smoke_$name.log 2>&1
  code=$?
  echo "$name exit $code tracebacks $(grep -ac Traceback $S/smoke_$name.log) iterations $(grep -ac 'Learning iteration' $S/smoke_$name.log)" >> $S/smoke_status.txt
}
legacy live_brs_him --task Isaac-Limx-SDBRS1-HIM-Blind-Rough-v0 --policy-type HIMPPO
echo DONE >> $S/smoke_status.txt
