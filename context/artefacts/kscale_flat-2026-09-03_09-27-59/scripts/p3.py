import sys
sys.path.insert(0,'/ws/tron1-rl-isaaclab-cozum/scripts/analysis')
from experiment_params import load_params, actuator_groups, action_terms
B='/ws/IsaacLab/logs/rsl_rl/kscale_flat/'
runs={'0828':'2026-08-28_04-50-51','arm0':'2026-09-03_09-27-59','base4':'2026-09-03_09-49-22','n1015':'2026-09-04_10-15-18','n1040':'2026-09-04_10-40-30'}
for k,v in runs.items():
    p=load_params(B+v+'/params/env.yaml')
    print('==',k,v)
    try:
        for a in actuator_groups(p): print('   ACT',a)
    except Exception as e: print('  actgrp err',e)
    try:
        for a in action_terms(p): print('   ACTION',a)
    except Exception as e: print('  act err',e)
