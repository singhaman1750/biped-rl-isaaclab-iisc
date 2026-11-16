"""Mean of every training scalar over the second half of each 400 iteration population window.

Run inside a training container, whose Python carries tensorboard:
    /workspace/isaaclab/isaaclab.sh -p tb_generation_bins.py <run_dir> [window_start ...]
With window starts given, prints 50 iteration bins from each start instead.
"""
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
ea.Reload()
groups = ("Episode_Reward", "Train", "Episode_Termination", "Curriculum", "Metrics", "Loss", "Policy")
tags = [t for t in ea.Tags()["scalars"] if t.split("/")[0] in groups and not t.endswith("/time")]
if len(sys.argv) > 2:
    lows = [lo + k for s in sys.argv[2:] for lo in [int(s)] for k in range(0, 400, 50)]
    width, offset = 50, 0
else:
    lows, width, offset = list(range(0, 8000, 400)), 400, 200
print("tag".ljust(44) + "".join(f"{b:>9d}" for b in lows))
for t in tags:
    ev = ea.Scalars(t)
    s = np.array([e.step for e in ev]); v = np.array([e.value for e in ev])
    row = []
    for lo in lows:
        m = (s >= lo + offset) & (s < lo + width)
        row.append(f"{v[m].mean():9.3f}" if m.any() else "        -")
    print(t[:44].ljust(44) + "".join(row))
