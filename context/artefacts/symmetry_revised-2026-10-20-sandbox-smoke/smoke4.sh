#!/bin/bash
# Sandbox only. play.py of the sandbox tree on copies of the two baselines' checkpoints, trained before the change,
# so that a checkpoint without symmetry is shown to load and run under a runner that carries it.
S=/tmp/symsandbox_v2_20261020
B=/workspace/isaaclab/logs/rsl_rl
true
for spec in "play_sf sf_tron_1a_flat/2026-10-20_05-24-20 model_2000.pt Isaac-Limx-SF-Identified-Blind-Rough-Play-Urdf-v0" \
            "play_quad quadruped_flat/2026-10-20_05-09-02 model_1000.pt Isaac-Quadruped-Blind-Rough-Play-v0"; do
  set -- $spec
  name=$1; run=$2; ckpt=$3; task=$4
  D=$S/playcheck/$run
  mkdir -p $D
  cd $S/playcheck
  PYTHONPATH=$S/environments:$S/himloco timeout 1500 /workspace/isaaclab/isaaclab.sh -p $S/scripts/rsl_rl/play.py \
    --task $task --num_envs 16 --video_length 60 --headless --checkpoint_path $D/$ckpt > $S/smoke_${name}2.log 2>&1
  echo "${name}2 exit $? tracebacks $(grep -ac Traceback $S/smoke_${name}2.log) loaded $(grep -ac 'Loading model checkpoint' $S/smoke_${name}2.log)" >> $S/smoke_status.txt
done
echo DONE5 >> $S/smoke_status.txt
