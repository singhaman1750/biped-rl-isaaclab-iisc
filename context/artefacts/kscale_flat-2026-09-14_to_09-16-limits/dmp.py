import numpy as np
BASE='/ws/IsaacLab/logs/rsl_rl/kscale_flat/'
RUNS=[('A 09-14','2026-09-14_10-20-50'),('B 09-15','2026-09-15_08-07-56'),('C 09-16','2026-09-16_04-52-20')]
def load(v): return np.load(BASE+v+'/data/42/dump.npy',allow_pickle=True).item()
