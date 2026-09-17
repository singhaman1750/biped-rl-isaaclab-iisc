import numpy as np
D=np.load("tb2.npy",allow_pickle=True).item()
NS=['0828','arm0','base4','n1015','n1040']
def s(n,tag,i,w=50):
    if tag not in D[n]: return float('nan')
    a=D[n][tag]; m=(a[:,0]>=i-w)&(a[:,0]<=i+w); return float(a[m,1].mean()) if m.any() else float('nan')
for tag,fmt in [('Train/mean_episode_length','8.1f'),('Loss/entropy','8.3f'),('Policy/mean_noise_std','8.4f'),('Train/mean_reward','8.2f')]:
    print(f"--- {tag}")
    print(f"{'it':>6s}"+"".join(f"{n:>9s}" for n in NS))
    for i in [0,250,500,1000,2000,3000,4000,5000,6000,8000,10000,13000,16000,20000,25000,29000]:
        v=[s(n,tag,i) for n in NS]
        print(f"{i:6d}"+"".join(f"{x:{fmt}} " for x in v))
    print()
