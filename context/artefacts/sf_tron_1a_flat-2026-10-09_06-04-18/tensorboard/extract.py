"""TensorBoard analysis of sf_tron_1a_flat/2026-10-09_06-04-18.

Run with
    PYTHONPATH=/ws/scratch/tron1_ft/pylib python3 /ws/scratch/tron1_ft/tb/extract.py

Outputs
    /ws/scratch/tron1_ft/tb/scalars_full.csv            full resolution, every iteration
    <ART>/scalars_every50.csv.gz                         raw values at iterations divisible by 50
    <ART>/plot_*.svg                                     hand written SVG line plots
    /ws/scratch/tron1_ft/tb/tables.md                    every table quoted in the report
    /ws/scratch/tron1_ft/tb/results.json                 the same numbers, machine readable
"""
import csv
import gzip
import json
import math
import os

import numpy as np
from scipy.optimize import curve_fit
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = "/ws/IsaacLab/logs/rsl_rl/sf_tron_1a_flat/2026-10-09_06-04-18"
EV = RUN + "/events.out.tfevents.1791526003.auk6000.175479.0"
SCR = "/ws/scratch/tron1_ft/tb"
ART = "/ws/tron1-rl-isaaclab-cozum/context/artefacts/sf_tron_1a_flat-2026-10-09_06-04-18/tensorboard"
os.makedirs(SCR, exist_ok=True)
os.makedirs(ART, exist_ok=True)

MARK = 12400          # user reported onset of walking on the feet (video step 310000 / 25)
ROLL = 50             # centred rolling mean window for every tabulated value
EP_LEN_MAX = 1000.0   # 20 s / 0.02 s
WEIGHTS = {           # from params/env.yaml of the run
    "keep_balance": 0.05, "rew_lin_vel_xy": 25.0, "rew_ang_vel_z": 7.5,
    "rew_keep_ankle_pitch_zero_in_air": 1.0, "rew_no_fly": 1.5, "pen_base_height": -5.0,
    "pen_lin_vel_z": -0.5, "pen_ang_vel_xy": -0.05, "pen_joint_torque": -8.0e-05,
    "pen_joint_accel": -5.0e-07, "pen_action_rate": -0.01, "pen_joint_pos_limits": -2.0,
    "pen_undesired_contacts": -0.5, "pen_action_smoothness": -0.075, "pen_flat_orientation": -1.0,
    "pen_feet_distance": -100.0, "pen_feet_regulation": -0.2, "pen_joint_vel_l2": -5.0e-05,
    "feet_air_time": 2.0, "feet_slide": -0.25,
}

out_md = []


def md(s=""):
    out_md.append(s)


# ---------------------------------------------------------------- 1. load
ea = EventAccumulator(EV, size_guidance={"scalars": 0, "tensors": 0, "histograms": 1,
                                         "images": 1, "audio": 1, "compressedHistograms": 1})
ea.Reload()
tags = sorted(ea.Tags()["scalars"])
raw = {}
for t in tags:
    ev = ea.Scalars(t)
    raw[t] = (np.array([e.step for e in ev], dtype=np.int64), np.array([e.value for e in ev], dtype=np.float64))

md("## T1. Scalar tags")
md()
md("| Tag | Points | First step | Last step | Unique steps |")
md("|---|---|---|---|---|")
for t in tags:
    s, v = raw[t]
    md(f"| `{t}` | {len(s)} | {s.min()} | {s.max()} | {len(np.unique(s))} |")
md()

# iteration indexed tags, the two /time tags are indexed by elapsed wall seconds instead
it_tags = [t for t in tags if not t.endswith("/time")]
N = int(max(raw[t][0].max() for t in it_tags)) + 1
S = {}
for t in it_tags:
    s, v = raw[t]
    a = np.full(N, np.nan)
    a[s] = v            # duplicate steps (none present) would keep the last write
    S[t] = a
elapsed = np.full(N, np.nan)
ts, _ = raw["Train/mean_reward/time"]
elapsed[: len(ts)] = ts     # k-th point of the /time tag is the k-th iteration's elapsed seconds
steps = np.arange(N)

# ---------------------------------------------------------------- 2. CSV export
cols = ["step", "elapsed_s"] + it_tags
with open(SCR + "/scalars_full.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for i in range(N):
        w.writerow([i, f"{elapsed[i]:.0f}"] + [repr(float(S[t][i])) for t in it_tags])
with gzip.open(ART + "/scalars_every50.csv.gz", "wt", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for i in range(0, N, 50):
        w.writerow([i, f"{elapsed[i]:.0f}"] + [f"{S[t][i]:.6g}" for t in it_tags])


# ---------------------------------------------------------------- helpers
def roll(a, k=ROLL):
    """Centred rolling mean over [i - k//2, i + k - k//2 - 1], truncated at the ends."""
    c = np.concatenate([[0.0], np.cumsum(a)])
    lo = np.clip(steps - k // 2, 0, N)
    hi = np.clip(steps + (k - k // 2), 0, N)
    return (c[hi] - c[lo]) / (hi - lo)


R = {t: roll(S[t]) for t in it_tags}
ER = lambda n: R["Episode_Reward/" + n]
L = R["Train/mean_episode_length"]
alive_len = L / EP_LEN_MAX                         # requested correction
alive_kb = ER("keep_balance") / WEIGHTS["keep_balance"]   # exact correction, see report


def fmt(x):
    if not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1000:
        return f"{x:.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.3f}"
    if ax >= 0.01:
        return f"{x:.4f}"
    return f"{x:.3g}"


def table(title, names, label_fn, rows, src):
    md(title)
    md()
    md("| Iteration | " + " | ".join(label_fn(n) for n in names) + " |")
    md("|---" * (len(names) + 1) + "|")
    for i in rows:
        md(f"| {i} | " + " | ".join(fmt(src[n][i]) for n in names) + " |")
    md()


CHK = [0, 100, 250, 500] + list(range(1000, 17001, 1000))
groups = {
    "Episode_Reward (positive weight)": [t for t in it_tags if t.startswith("Episode_Reward/")
                                         and WEIGHTS[t.split("/")[1]] > 0],
    "Episode_Reward (negative weight), part 1": [t for t in it_tags if t.startswith("Episode_Reward/")
                                                 and WEIGHTS[t.split("/")[1]] < 0][:7],
    "Episode_Reward (negative weight), part 2": [t for t in it_tags if t.startswith("Episode_Reward/")
                                                 and WEIGHTS[t.split("/")[1]] < 0][7:],
    "Episode_Termination, Train, Policy, Metrics": [t for t in it_tags if t.split("/")[0] in
                                                    ("Episode_Termination", "Train", "Policy", "Metrics")],
    "Curriculum": [t for t in it_tags if t.startswith("Curriculum/")],
    "Loss and Perf": [t for t in it_tags if t.split("/")[0] in ("Loss", "Perf")],
}
md("## T2. Checkpoint table, centred 50 iteration rolling mean")
md()
for g, names in groups.items():
    table(f"### {g}", names, lambda n: n.split("/", 1)[1], CHK, R)

# ---------------------------------------------------------------- 4. change points
KEY = {
    "ep_len": "Train/mean_episode_length",
    "base_contact": "Episode_Termination/base_contact",
    "pen_base_height": "Episode_Reward/pen_base_height",
    "pen_undesired_contacts": "Episode_Reward/pen_undesired_contacts",
    "rew_lin_vel_xy": "Episode_Reward/rew_lin_vel_xy",
    "rew_ang_vel_z": "Episode_Reward/rew_ang_vel_z",
    "feet_air_time": "Episode_Reward/feet_air_time",
    "rew_no_fly": "Episode_Reward/rew_no_fly",
    "pen_feet_distance": "Episode_Reward/pen_feet_distance",
    "pen_feet_regulation": "Episode_Reward/pen_feet_regulation",
    "ankle_pitch_air": "Episode_Reward/rew_keep_ankle_pitch_zero_in_air",
    "keep_balance": "Episode_Reward/keep_balance",
    "noise_std": "Policy/mean_noise_std",
    "mean_reward": "Train/mean_reward",
}


def block(a, b):
    n = len(a) // b
    return a[: n * b].reshape(n, b).mean(1), np.arange(n) * b + (b - 1) / 2.0


def binseg(y, k, minseg=4):
    """Binary segmentation, L2 mean shift cost. Returns [(index, cost_reduction)] in discovery order."""
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    cost = lambda a, b: (c2[b] - c2[a]) - (c1[b] - c1[a]) ** 2 / (b - a)
    segs, cps = [(0, len(y))], []
    for _ in range(k):
        best = None
        for (a, b) in segs:
            if b - a < 2 * minseg:
                continue
            base = cost(a, b)
            for m in range(a + minseg, b - minseg + 1):
                g = base - cost(a, m) - cost(m, b)
                if best is None or g > best[0]:
                    best = (g, a, b, m)
        if best is None:
            break
        g, a, b, m = best
        segs.remove((a, b))
        segs += [(a, m), (m, b)]
        cps.append((m, g))
    return cps


def logistic(x, a, b, x0, w):
    return a + b / (1.0 + np.exp(-(x - x0) / w))


res = {"change_points": {}}
md("## T3. Change points")
md()
md("Global segmentation uses binary segmentation with an L2 mean shift cost on 50 iteration block means, "
   "five change points per series, listed in iteration order with the share of the total sum of squares each one removes. "
   "The 10000 to 14000 window uses two estimators on the raw series, a four parameter logistic fitted to 10 iteration block means, "
   "reporting its midpoint and its 10 to 90 percent rise interval (midpoint plus or minus 2.197 w), and a step contrast, "
   "the iteration t maximising |mean(t, t+300) minus mean(t-300, t)| divided by the pooled standard deviation.")
md()
md("| Series | Global change points (iteration, share of SS) | Logistic midpoint | 10 to 90 percent interval | Logistic amplitude | Logistic R^2 | Step contrast t | Contrast before to after |")
md("|---|---|---|---|---|---|---|---|")
for k, t in KEY.items():
    y = S[t]
    yb, xb = block(y, 50)
    sst = float(np.sum((yb - yb.mean()) ** 2))
    cps = binseg(yb, 5)
    cp_txt = ", ".join(f"{int(m * 50)} ({g / sst * 100:.0f}%)" for m, g in sorted(cps))
    # window estimators
    w0, w1 = 10000, 14000
    yw, xw = block(y[w0:w1], 10)
    xw = xw + w0
    p0 = [yw[:20].mean(), yw[-20:].mean() - yw[:20].mean(), 12400.0, 200.0]
    try:
        p, _ = curve_fit(logistic, xw, yw, p0=p0, maxfev=20000,
                         bounds=([-np.inf, -np.inf, w0, 10.0], [np.inf, np.inf, w1, 3000.0]))
        r2 = 1 - np.sum((yw - logistic(xw, *p)) ** 2) / np.sum((yw - yw.mean()) ** 2)
        lo, hi = p[2] - 2.197 * p[3], p[2] + 2.197 * p[3]
    except Exception:
        p, r2, lo, hi = [np.nan] * 4, np.nan, np.nan, np.nan
    H = 300
    best = (0, None, None, None)
    for tt in range(w0, w1 + 1, 10):
        a, b = y[tt - H:tt], y[tt:tt + H]
        z = (b.mean() - a.mean()) / math.sqrt(0.5 * (a.var() + b.var()) + 1e-30)
        if abs(z) > abs(best[0]):
            best = (z, tt, a.mean(), b.mean())
    res["change_points"][k] = {"global": [(int(m * 50), g / sst) for m, g in sorted(cps)],
                               "logistic": [float(q) for q in p], "r2": float(r2),
                               "rise": [float(lo), float(hi)], "contrast": [best[1], best[0], best[2], best[3]]}
    md(f"| {k} | {cp_txt} | {p[2]:.0f} | {lo:.0f} to {hi:.0f} | {fmt(p[1])} | {r2:.3f} | {best[1]} (z {best[0]:.2f}) | {fmt(best[2])} to {fmt(best[3])} |")
md()

dense = list(range(11000, 13501, 100))
kl = list(KEY.items())
for part, sub in (("part 1", kl[:7]), ("part 2", kl[7:])):
    md(f"### Dense table 11000 to 13500, {part}, centred 50 iteration rolling mean")
    md()
    md("| Iteration | " + " | ".join(k for k, _ in sub) + " |")
    md("|---" * (len(sub) + 1) + "|")
    for i in dense:
        md(f"| {i} | " + " | ".join(fmt(R[t][i]) for _, t in sub) + " |")
    md()

# ---------------------------------------------------------------- 5. reward budget and implied quantities
BUD = [2000, 6000, 10000, 13000, 16000]
terms = [t.split("/")[1] for t in it_tags if t.startswith("Episode_Reward/")]
res["budget"] = {}
md("## T4. Reward budget, rate per second of the weighted term, centred 50 iteration rolling mean")
md()
for i in BUD:
    vals = {n: float(ER(n)[i]) for n in terms}
    pos = sum(v for v in vals.values() if v > 0)
    neg = sum(v for v in vals.values() if v < 0)
    res["budget"][i] = {"terms": vals, "pos": pos, "neg": neg}
    md(f"### Iteration {i}")
    md()
    md(f"Total positive {pos:.3f} per s, total negative {neg:.3f} per s, net {pos + neg:.3f} per s, "
       f"net times 20 s {20 * (pos + neg):.1f}, Train/mean_reward {R['Train/mean_reward'][i]:.1f}, "
       f"mean episode length {L[i]:.1f} steps.")
    md()
    md("| Term | Weight | Rate per s | Share of its sign group |")
    md("|---|---|---|---|")
    for n, v in sorted(vals.items(), key=lambda kv: -abs(kv[1])):
        sh = v / pos if v > 0 else v / neg
        md(f"| {n} | {WEIGHTS[n]:g} | {fmt(v)} | {sh * 100:.1f}% |")
    md()

# implied per step means: f_bar = ER / (w * alive), alive from mean_episode_length or from keep_balance
IMPL = {
    "pen_base_height": "mean squared height error (m^2)",
    "pen_undesired_contacts": "mean non foot bodies in contact per step",
    "rew_no_fly": "mean no_fly value per step (1 single, minus 5 none, 0 double)",
    "feet_air_time": "mean feet_air_time value per step (s)",
    "pen_feet_distance": "mean feet distance hinge (m)",
    "pen_feet_regulation": "mean feet_regulation value per step (m^2/s^2)",
    "rew_keep_ankle_pitch_zero_in_air": "mean ankle pitch kernel per step",
    "pen_lin_vel_z": "mean squared base vertical velocity (m^2/s^2)",
    "pen_flat_orientation": "mean squared projected gravity xy",
    "rew_lin_vel_xy": "mean linear tracking kernel exp(-e^2/std^2)",
    "rew_ang_vel_z": "mean yaw tracking kernel exp(-e^2/std^2)",
}
imp = {}
for n in IMPL:
    imp[n + "|len"] = ER(n) / (WEIGHTS[n] * alive_len)
    imp[n + "|kb"] = ER(n) / (WEIGHTS[n] * alive_kb)
imp["height_rms|len"] = np.sqrt(np.clip(imp["pen_base_height|len"], 0, None))
imp["height_rms|kb"] = np.sqrt(np.clip(imp["pen_base_height|kb"], 0, None))
imp["vz_rms|kb"] = np.sqrt(np.clip(imp["pen_lin_vel_z|kb"], 0, None))
imp["tilt_deg|kb"] = np.degrees(np.arcsin(np.sqrt(np.clip(imp["pen_flat_orientation|kb"], 0, 1))))
imp["alive_len"] = alive_len
imp["alive_kb"] = alive_kb
res["implied"] = {}
IMP_ROWS = [2000, 6000, 10000, 11000, 12000, 12400, 13000, 14000, 16000, 17000]
md("## T5. Implied per step quantities")
md()
md("Each entry is the Episode_Reward rate divided by the weight and by the alive fraction. "
   "Column suffix len uses alive = Train/mean_episode_length / 1000, column suffix kb uses alive = Episode_Reward/keep_balance / 0.05.")
md()
cols5 = ["alive_len", "alive_kb", "pen_base_height|len", "pen_base_height|kb", "height_rms|len", "height_rms|kb",
         "pen_undesired_contacts|len", "pen_undesired_contacts|kb"]
cols6 = ["rew_no_fly|len", "rew_no_fly|kb", "feet_air_time|len", "feet_air_time|kb",
         "pen_feet_distance|kb", "pen_feet_regulation|kb", "rew_keep_ankle_pitch_zero_in_air|kb"]
cols7 = ["pen_lin_vel_z|kb", "vz_rms|kb", "pen_flat_orientation|kb", "tilt_deg|kb",
         "rew_lin_vel_xy|kb", "rew_ang_vel_z|kb"]
for cs in (cols5, cols6, cols7):
    md("| Iteration | " + " | ".join(c for c in cs) + " |")
    md("|---" * (len(cs) + 1) + "|")
    for i in IMP_ROWS:
        md(f"| {i} | " + " | ".join(fmt(imp[c][i]) for c in cs) + " |")
        res["implied"].setdefault(i, {}).update({c: float(imp[c][i]) for c in cs})
    md()
# no_fly decomposition bounds and feet air time per second
md("no_fly bounds. With f the kb mean per step, assuming no double support gives single fraction (f+5)/6 and no-contact fraction (1-f)/6, "
   "assuming no flight phase gives single fraction f and double fraction 1-f (valid only for f >= 0). feet_air_time per second is f/0.02.")
md()
md("| Iteration | f no_fly | single if no double support | none if no double support | single if no flight | feet_air_time sum per s |")
md("|---|---|---|---|---|---|")
for i in IMP_ROWS:
    f = imp["rew_no_fly|kb"][i]
    fa = imp["feet_air_time|kb"][i]
    md(f"| {i} | {fmt(f)} | {fmt((f + 5) / 6)} | {fmt((1 - f) / 6)} | {fmt(f) if f >= 0 else 'n/a'} | {fmt(fa / 0.02)} |")
md()

# ---------------------------------------------------------------- 6. curricula
md("## T6. Curricula")
md()
cur = [t for t in it_tags if t.startswith("Curriculum/")]
res["curriculum"] = {}
md("| Tag | Value at 0 | First iteration differing from 0 | Number of value changes | Iterations of the first eight changes | Final value |")
md("|---|---|---|---|---|---|")
for t in cur:
    a = S[t]
    if t.endswith("terrain_levels"):
        d = np.nonzero(np.abs(a - a[0]) > 1e-9)[0]
        md(f"| `{t}` | {fmt(a[0])} | {d[0] if len(d) else 'never'} | continuous | n/a | {fmt(a[-1])} |")
        res["curriculum"][t] = {"first_change": int(d[0]) if len(d) else None}
        continue
    ch = np.nonzero(np.abs(np.diff(a)) > 1e-7)[0] + 1
    md(f"| `{t}` | {fmt(a[0])} | {ch[0] if len(ch) else 'never'} | {len(ch)} | "
       f"{', '.join(f'{c} ({fmt(a[c])})' for c in ch[:8])} | {fmt(a[-1])} |")
    res["curriculum"][t] = {"changes": [(int(c), float(a[c])) for c in ch]}
md()
md("Curriculum values (raw, not averaged) at checkpoints")
md()
md("| Iteration | " + " | ".join(t.split("/")[1] for t in cur) + " |")
md("|---" * (len(cur) + 1) + "|")
for i in CHK + [12400, N - 1]:
    md(f"| {i} | " + " | ".join(fmt(S[t][i]) for t in cur) + " |")
md()

# ---------------------------------------------------------------- 7. SVG plots
BLUE, ORANGE, TXT, TXT2, GRID, SURF, MARKC = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"


def nice_ticks(lo, hi, n=5):
    span = hi - lo if hi > lo else 1.0
    raw_step = span / n
    mag = 10 ** math.floor(math.log10(raw_step))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw_step), default=10 * mag)
    t0 = math.ceil(lo / step) * step
    out = []
    v = t0
    while v <= hi + 1e-12:
        out.append(round(v, 10))
        v += step
    return out


def svg_plot(fname, title, ylabel, series, note):
    """series: list of (label, x, y, colour). One y axis only."""
    W, H, ml, mr, mt, mb = 760, 330, 70, 20, 46, 52
    pw, ph = W - ml - mr, H - mt - mb
    ys = np.concatenate([s[2][np.isfinite(s[2])] for s in series])
    lo, hi = float(ys.min()), float(ys.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad
    xmax = N - 1
    X = lambda x: ml + pw * x / xmax
    Y = lambda y: mt + ph * (1 - (y - lo) / (hi - lo))
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="{ml}" y="20" font-size="15" fill="{TXT}">{title}</text>',
         f'<text x="{ml}" y="36" font-size="11" fill="{TXT2}">{note}</text>']
    for v in nice_ticks(lo, hi):
        e.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        e.append(f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{TXT2}">{v:g}</text>')
    for v in range(0, xmax + 1, 2000):
        e.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{mt + ph}" y2="{mt + ph + 4}" stroke="{TXT2}"/>')
        e.append(f'<text x="{X(v):.1f}" y="{mt + ph + 17}" text-anchor="middle" fill="{TXT2}">{v}</text>')
    e.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" stroke="{TXT2}"/>')
    e.append(f'<line x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + ph}" stroke="{TXT2}"/>')
    e.append(f'<text x="{ml + pw / 2}" y="{H - 10}" text-anchor="middle" fill="{TXT}">PPO iteration</text>')
    e.append(f'<text transform="translate(16 {mt + ph / 2}) rotate(-90)" text-anchor="middle" fill="{TXT}">{ylabel}</text>')
    e.append(f'<line x1="{X(MARK):.1f}" x2="{X(MARK):.1f}" y1="{mt}" y2="{mt + ph}" stroke="{MARKC}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    e.append(f'<text x="{X(MARK) + 4:.1f}" y="{mt + 12}" fill="{TXT2}" font-size="11">12400, feet walking reported</text>')
    for j, (lab, x, y, col) in enumerate(series):
        ok = np.isfinite(y)
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x[ok], y[ok]))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        if len(series) > 1:
            lx = ml + pw - 230 + j * 120
            e.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{mt - 4}" y2="{mt - 4}" stroke="{col}" stroke-width="2"/>')
            e.append(f'<text x="{lx + 22}" y="{mt}" fill="{TXT}" font-size="11">{lab}</text>')
    e.append("</svg>")
    p = os.path.join(ART, fname)
    with open(p, "w") as f:
        f.write("\n".join(e))
    return p, os.path.getsize(p)


def b50(a):
    y, x = block(a, 50)
    return x, y


NOTE = "50 iteration block means"
plots = []
x, y = b50(S["Train/mean_episode_length"])
plots.append(svg_plot("plot_episode_length.svg", "Mean episode length", "control steps (max 1000)",
                      [("episode length", x, y, BLUE)], NOTE))
x, y = b50(np.sqrt(np.clip(S["Episode_Reward/pen_base_height"] /
                           (WEIGHTS["pen_base_height"] * S["Episode_Reward/keep_balance"] / 0.05), 0, None)))
plots.append(svg_plot("plot_base_height_rms_error.svg", "Implied RMS base height error from pen_base_height",
                      "sqrt(mean (h - 0.75)^2) (m)", [("rms error", x, y, BLUE)],
                      NOTE + ", alive fraction from keep_balance / 0.05"))
x, y = b50(S["Episode_Reward/pen_undesired_contacts"] /
           (WEIGHTS["pen_undesired_contacts"] * S["Episode_Reward/keep_balance"] / 0.05))
plots.append(svg_plot("plot_undesired_contacts.svg", "Implied non foot bodies in contact per step",
                      "bodies in contact per step", [("contacts", x, y, BLUE)],
                      NOTE + ", alive fraction from keep_balance / 0.05"))
x, y1 = b50(S["Episode_Reward/rew_lin_vel_xy"])
_, y2 = b50(S["Episode_Reward/rew_ang_vel_z"])
plots.append(svg_plot("plot_tracking.svg", "Velocity tracking rewards", "weighted rate per s",
                      [("rew_lin_vel_xy", x, y1, BLUE), ("rew_ang_vel_z", x, y2, ORANGE)], NOTE))
x, y1 = b50(S["Episode_Reward/rew_no_fly"])
_, y2 = b50(S["Episode_Reward/feet_air_time"])
plots.append(svg_plot("plot_foot_contact_rewards.svg", "Foot contact rewards", "weighted rate per s",
                      [("rew_no_fly", x, y1, BLUE), ("feet_air_time", x, y2, ORANGE)], NOTE))
x, y = b50(S["Policy/mean_noise_std"])
plots.append(svg_plot("plot_noise_std.svg", "Policy mean action noise std", "std", [("noise std", x, y, BLUE)], NOTE))
md("## T7. Plots")
md()
for p, sz in plots:
    md(f"- `{p}` {sz} bytes")
md()

with open(SCR + "/tables.md", "w") as f:
    f.write("\n".join(out_md) + "\n")
with open(SCR + "/results.json", "w") as f:
    json.dump(res, f, indent=1, default=float)
print("done", N, "iterations,", len(tags), "tags")
