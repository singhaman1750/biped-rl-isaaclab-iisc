"""The true sole polygon separation against the ankle frame separation the live term measures.

pen_feet_distance calls rewards.py:1038 feet_distance, which measures the distance between the two
foot LINK FRAME ORIGINS, and rewards.py:1138 feet_distance_v2 documents that on a sole footed robot
this is the ankle and not the foot.  The two diverge the moment a foot yaws, and the toe-in these
policies carry is exactly that.  This program measures both from the dump, using the same vertex
against edge sweep _footprint_separation performs at rewards.py:1091, so the gap the proposed
second hinge would charge is a measurement of these runs rather than an estimate.
"""
import numpy as np, common
from frames import quat_rot

SOLE = np.array([
    [-0.0423,+0.0430,+0.0138],[-0.0362,+0.0430,-0.1572],[-0.0326,+0.0430,-0.1712],
    [-0.0258,+0.0430,-0.1805],[-0.0108,+0.0430,-0.1889],[+0.0051,+0.0430,-0.1899],
    [+0.0213,+0.0430,-0.1841],[+0.0326,+0.0430,-0.1712],[+0.0362,+0.0430,-0.1572],
    [+0.0423,+0.0430,+0.0138],[+0.0363,+0.0430,+0.0200],[-0.0363,+0.0430,+0.0200]])

def seg_dist(P, Q):
    """Minimum planar distance between two convex polygons, vertex against edge both ways."""
    best = None
    for near, far in ((P, Q), (Q, P)):
        edge = np.roll(far, -1, axis=-2) - far
        delta = near[..., :, None, :] - far[..., None, :, :]
        sc = (delta*edge[..., None, :, :]).sum(-1) / np.maximum((edge*edge).sum(-1), 1e-12)[..., None, :]
        foot = delta - np.clip(sc, 0, 1)[..., None]*edge[..., None, :, :]
        span = np.linalg.norm(foot, axis=-1).min(axis=(-2, -1))
        best = span if best is None else np.minimum(best, span)
    return best

print(f"{'run':12}{'ankle sep':>11}{'footprint sep':>15}{'gap':>9}{'below 0.19':>12}{'below 0.05':>12}{'below 0.02':>12}{'touching':>10}")
for run in common.RUNS:
    d = common.load(run)
    fq = d["feet_quaternions"]; sep = d["feet_distance"]        # (T,E,3) base yaw frame
    T, E = sep.shape[0], sep.shape[1]
    # sample to keep the polygon sweep tractable, every 4th step across all envs
    sl = slice(None, None, 4)
    pts = []
    for fi in (0, 1):
        q = fq[sl, :, fi, :]
        R = np.stack([quat_rot(q, v) for v in np.eye(3)], axis=-1)   # (...,3,3) columns are axes
        pts.append(np.einsum("...ij,pj->...pi", R, SOLE))            # (T,E,P,3) world offsets
    # foot 0 at the origin, foot 1 displaced by the recorded separation, in a common planar frame
    P = pts[0][..., :2]
    Q = pts[1][..., :2] + sep[sl, :, None, :2]
    ankle = np.linalg.norm(sep[sl, :, :2], axis=-1)
    fp = seg_dist(P, Q)
    print(f"{run:12}{ankle.mean():>11.4f}{fp.mean():>15.4f}{(ankle-fp).mean():>9.4f}"
          f"{100*np.mean(fp<0.19):>11.1f}%{100*np.mean(fp<0.05):>11.1f}%{100*np.mean(fp<0.02):>11.1f}%"
          f"{100*np.mean(fp<=1e-6):>9.1f}%")
print("\npercentiles of the footprint separation, metres")
print(f"{'run':12}" + "".join(f"{f'p{p}':>9}" for p in (1,5,25,50,75,99)))
for run in common.RUNS:
    d = common.load(run)
    fq = d["feet_quaternions"]; sep = d["feet_distance"]
    sl = slice(None, None, 4)
    pts = []
    for fi in (0, 1):
        q = fq[sl, :, fi, :]
        R = np.stack([quat_rot(q, v) for v in np.eye(3)], axis=-1)
        pts.append(np.einsum("...ij,pj->...pi", R, SOLE))
    fp = seg_dist(pts[0][..., :2], pts[1][..., :2] + sep[sl, :, None, :2])
    print(f"{run:12}" + "".join(f"{np.percentile(fp,p):>9.4f}" for p in (1,5,25,50,75,99)))
