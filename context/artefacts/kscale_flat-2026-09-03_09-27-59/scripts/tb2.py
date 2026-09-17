import sys, numpy as np
from tensorboard.backend.event_processing import event_accumulator
base="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
runs={"0828":"2026-08-28_04-50-51","arm0":"2026-09-03_09-27-59","base4":"2026-09-03_09-49-22","n1015":"2026-09-04_10-15-18","n1040":"2026-09-04_10-40-30"}
DATA={}
for name,d in runs.items():
    ea=event_accumulator.EventAccumulator(f"{base}/{d}", size_guidance={'scalars':0})
    ea.Reload()
    tags=ea.Tags()['scalars']
    DATA[name]={t:np.array([(s.step,s.value) for s in ea.Scalars(t)]) for t in tags}
np.save("/root/.claude/jobs/b40fca2a/tmp/tb2.npy", DATA, allow_pickle=True)
for name in runs:
    print(name, "tags:", len(DATA[name]), "maxstep:", max(v[-1,0] for v in DATA[name].values()))
print()
print(sorted(DATA['base4'].keys()))
