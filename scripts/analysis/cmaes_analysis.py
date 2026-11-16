"""CMA-ES diagnostics of a co-optimisation run, read from its model checkpoints.

Every checkpoint carries the pickled search strategy under copt.design_generator.es and the
per generation statistics under copt.cmaes_history (copt_on_policy_runner.py save). The
strategy is read through an adapter, so the pycma search of CMAESDesignGenerator and the
mixed continuous and categorical CatCMAwM search of CatCMAESDesignGenerator give the same
snapshot and share every plot. The plots and their choice are explained in
/ws/context/cmaes.md section 0.11.

The full covariance exists only in the checkpoints, one every save_interval iterations, so
the covariance figures, the principal axes, the evolution path and the population plots
are sampled at the checkpoints, while the statistics logged every generation are plotted
for every generation.

Usage:
    python scripts/analysis/cmaes_analysis.py <exp_dir> [--out DIR] [--show] [--fps 2]

Run it with the Isaac Lab python, which has torch, cma, cmaes and matplotlib. Writes the
cmaes_*.png figures and cmaes_covariance_animation.gif (and .mp4 when an ffmpeg is found) to
<exp_dir>/designs/plots/ unless --out is given, and cmaes_checkpoint_summary.csv beside the
other design tables in <exp_dir>/designs/. --show opens the covariance animation in a window.
"""

import argparse
import csv
import glob
import os
import pickle
import re
import textwrap

import numpy as np
import torch

LEGEND = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)
SPLIT_HALF_STOP = 0.3  # stopping threshold of plans/copt_improvement.md section 5.4 step 5
SNR_BETA = 0.1  # beta_m of Nomura, Akimoto and Ono (GECCO 2023), section 4


# --- loading -------------------------------------------------------------------------


def checkpoints(exp_dir):
    paths = glob.glob(os.path.join(exp_dir, "model_*.pt"))
    return sorted(paths, key=lambda p: int(re.search(r"model_(\d+)\.pt$", p).group(1)))


def load_copt(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    return c["iter"], c.get("copt")


def snapshot(es, last_solutions):
    """One strategy state in the normalised coordinates of the unit box.

    cov is the covariance actually sampled, sigma^2 S C S for pycma with S its sigma_vec
    and sigma^2 A C A for CatCMAwM, and C_axis_ratio the ratio the runner logs.
    """
    name = type(es).__name__
    if name == "CatCMAwM":
        a = np.asarray(es._A, dtype=float)
        C = np.asarray(es._C, dtype=float)
        cov = es._sigma ** 2 * a[:, None] * C * a[None, :]
        n = int(es._Nco)
        s: dict = dict(kind="CatCMAwM", iteration=int(es._g), sigma=float(es._sigma),
                 mean_geno=np.asarray(es._mean, dtype=float)[:n], scaling=a[:n],
                 path_ratio=float(np.linalg.norm(es._p_sigma) / es._chi_n))
        s["C_axis_ratio"] = float(np.sqrt(np.linalg.eigvalsh(cov).max() / np.linalg.eigvalsh(cov).min()))
        s["q"] = np.asarray(es._q, dtype=float) if getattr(es, "_use_categorical", False) else None
        s["K"] = np.asarray(es._K) if s["q"] is not None else None
        pop = [np.asarray(getattr(x, "x", x), dtype=float)[:n] for x in (last_solutions or [])]
        cats = [np.asarray(getattr(x, "c")).argmax(-1) for x in (last_solutions or []) if hasattr(x, "c")]
        s["stds"] = np.sqrt(np.diag(cov))[:n]
        s["xfavorite"] = np.clip(s["mean_geno"], 0.0, 1.0)
    else:
        C = np.asarray(es.sm.C, dtype=float)
        sv = np.asarray(es.sigma_vec.scaling, dtype=float) * np.ones(es.N)
        cov = es.sigma ** 2 * sv[:, None] * C * sv[None, :]
        n = int(es.N)
        chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        ps = getattr(es.adapt_sigma, "ps", None)
        s: dict = dict(kind="pycma", iteration=int(es.countiter), sigma=float(es.sigma),
                 mean_geno=np.asarray(es.mean, dtype=float), scaling=sv,
                 path_ratio=float(np.linalg.norm(ps) / chi) if ps is not None else np.nan,
                 C_axis_ratio=float(es.D.max() / es.D.min()), q=None, K=None)
        pop = [np.asarray(x, dtype=float) for x in (last_solutions or [])]
        cats = []
        s["stds"] = np.asarray(es.result.stds, dtype=float)
        s["xfavorite"] = np.asarray(es.result.xfavorite, dtype=float)
    cov = cov[:n, :n]
    d = np.sqrt(np.diag(cov))
    ev, vec = np.linalg.eigh(cov)
    order = ev.argsort()[::-1]
    s.update(cov=cov, corr=cov / np.outer(d, d), axis_len=np.sqrt(ev[order]), axis_vec=vec[:, order],
             population=np.array(pop) if pop else None, categories=np.array(cats) if cats else None)
    return s


def load_run(exp_dir):
    """Snapshots of every checkpoint after the hand-over, and the longest history found."""
    snaps, history = [], []
    for path in checkpoints(exp_dir):
        it, copt = load_copt(path)
        if not copt or not copt.get("copt_started"):
            continue
        if len(copt.get("cmaes_history") or []) > len(history):
            history = copt["cmaes_history"]
        dg = copt["design_generator"]
        es = pickle.loads(dg["es"])
        s = snapshot(es, dg.get("last_solutions"))
        if s["iteration"] < 1:
            continue
        s["checkpoint"] = it
        snaps.append(s)
    if not history:
        raise SystemExit(f"no CMA-ES history in the checkpoints of {exp_dir}")
    return snaps, history


# --- derived quantities ----------------------------------------------------------------


def series(history, key):
    return np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in history])


def param_names(history):
    return [k[len("mean/"):] for k in history[0] if k.startswith("mean/")]


def generation_of(history):
    return {int(r["cma_iteration"]): int(r["generation"]) for r in history}


def infer_ranges(snaps, history, names):
    """Lower bound and span of each parameter, from the runner's own denormalisation.

    search_statistics logs std = span * stds and mean = lo + span * xfavorite, so a
    checkpoint and the history row of the same CMA iteration give both exactly.
    """
    rows = {int(r["cma_iteration"]): r for r in history}
    lo, span = [], []
    for s in snaps:
        r = rows.get(s["iteration"])
        if r is None:
            continue
        sp = np.array([float(r[f"std/{k}"]) for k in names]) / np.maximum(s["stds"], 1e-300)
        lo.append(np.array([float(r[f"mean/{k}"]) for k in names]) - sp * np.clip(s["xfavorite"], 0, 1))
        span.append(sp)
    if not span:
        raise SystemExit("no checkpoint shares a CMA iteration with the history")
    return np.median(lo, axis=0), np.median(span, axis=0)


def mean_shift_snr(history, names, beta=SNR_BETA):
    """Signal to noise ratio of the mean update, eqs. 11 and 13b of Nomura et al. (2023).

    The update is whitened by the per coordinate standard deviation of the distribution
    that produced it, a diagonal approximation of Sigma^(-1/2), since the full covariance
    is stored only at the checkpoints.
    """
    m = np.stack([series(history, f"mean/{k}") for k in names], 1)
    sd = np.stack([series(history, f"std/{k}") for k in names], 1)
    E, V, out = np.zeros(len(names)), 0.0, [np.nan]
    for t in range(1, len(history)):
        d = (m[t] - m[t - 1]) / np.maximum(sd[t - 1], 1e-300)
        E = (1 - beta) * E + beta * d
        V = (1 - beta) * V + beta * float(d @ d)
        e2 = float(E @ E)
        out.append((e2 - beta / (2 - beta) * V) / max(V - e2, 1e-300))
    return np.array(out)


def evenly_spaced(n, k=4):
    return sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist()))


# --- plotting helpers ------------------------------------------------------------------


def robust_ylim(ax, values, k=20.0):
    """Clip the y axis to the bulk of the data and name the points left outside it.

    Only values beyond k interquartile ranges of the quartiles are left out, so a slow
    early rise stays on the plot while a single blown up evaluation does not flatten it.
    """
    v = np.concatenate([np.asarray(x, float).ravel() for x in values])
    v = v[np.isfinite(v)]
    med = np.median(v)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = (q3 - q1) or (np.std(v) or 1.0)
    lo, hi = max(v.min(), q1 - k * iqr), min(v.max(), q3 + k * iqr)
    pad = 0.08 * (hi - lo or 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    off = v[(v < lo) | (v > hi)]
    if off.size:
        ax.text(0.01, 0.02, f"{off.size} value(s) off scale, extreme {off[np.argmax(np.abs(off - med))]:.4g}",
                transform=ax.transAxes, fontsize=7, color="0.3")


def log_ticks(ax):
    """Label the minor ticks of a log axis that spans less than about a decade."""
    lo, hi = ax.get_ylim()
    if lo > 0 and hi / lo < 30:
        def fmt_fn(v, _):
            lead = int(round(v / 10 ** np.floor(np.log10(v))))
            return f"{v:.2g}" if lead in (1, 2, 3, 5) else ""

        fmt = mticker.FuncFormatter(fmt_fn)
        ax.yaxis.set_minor_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.tick_params(axis="y", which="minor", labelsize=7)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def wrap(names, width=12):
    return [textwrap.fill(n.replace("_", " "), width) for n in names]


def heatmap(ax, M, labels_x, labels_y, cmap, vmin, vmax, fmt, title, ylabels=True):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels_x)), labels_x, fontsize=7)
    ax.set_yticks(range(len(labels_y)), labels_y if ylabels else [], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.6 * max(abs(vmin), abs(vmax)) else "black")
    ax.set_title(title, fontsize=9)
    return im


# --- plots -----------------------------------------------------------------------------


def plot_fitness(history, out):
    g = series(history, "generation")
    mean, sd = series(history, "fitness_mean"), series(history, "fitness_std")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(g, mean - sd, mean + sd, alpha=0.2, color="C0", label="mean ± 1 std")
    ax.plot(g, mean, "-o", ms=3, color="C0", label="mean")
    ax.plot(g, series(history, "fitness_median"), "--", color="C1", label="median")
    ax.plot(g, series(history, "fitness_max"), "-", color="C2", label="max")
    ax.plot(g, series(history, "fitness_min"), "-", color="C3", label="min")
    ax.plot(g, np.fmax.accumulate(series(history, "fitness_max")), ":", color="k", label="best so far")
    robust_ylim(ax, [mean, series(history, "fitness_max"), series(history, "fitness_min")])
    ax.set(xlabel="generation", ylabel="fitness (evaluation return)", title="Fitness of the population")
    ax.grid(alpha=0.3)
    ax.legend(**LEGEND)
    save(fig, out, "cmaes_fitness.png")


def plot_step_size(history, snaps, gen, out):
    g = series(history, "generation")
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axs[0].semilogy(g, series(history, "sigma"), "-o", ms=3, label="step size sigma")
    axs[0].set(ylabel="sigma (normalised units)", title="Step size, axis ratio and evolution path")
    axs[1].semilogy(g, series(history, "axis_ratio"), "-o", ms=3, color="C1", label="axis ratio, logged")
    axs[1].set_ylabel("axis ratio, max/min axis")
    gs = [gen.get(s["iteration"], np.nan) for s in snaps]
    axs[2].plot(gs, [s["path_ratio"] for s in snaps], "s-", color="C2", label="||p_sigma|| / E||N(0,I)||, checkpoints")
    axs[2].axhline(1.0, ls="--", color="k", lw=1, label="random selection reference")
    axs[2].set(xlabel="generation", ylabel="path length ratio")
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
        ax.legend(**LEGEND)
    save(fig, out, "cmaes_step_size.png")


def plot_coordinate(history, snaps, gen, names, lo, span, out):
    """One row per coordinate, its value on the left and the search standard deviation on the right.

    The value panel carries the distribution mean with its one standard deviation window, the
    best design of each generation, the unclipped internal mean read from the checkpoints, and
    the search bounds.
    """
    g = series(history, "generation")
    gs = [gen.get(s["iteration"], np.nan) for s in snaps]
    fig, axs = plt.subplots(len(names), 2, figsize=(13, 2.8 * len(names)), sharex=True, squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.4, 1]))
    for i, k in enumerate(names):
        m, sd, best = series(history, f"mean/{k}"), series(history, f"std/{k}"), series(history, f"best/{k}")
        ax = axs[i, 0]
        ax.fill_between(g, m - sd, m + sd, alpha=0.2, color="C0", label="mean ± 1 std of the distribution")
        ax.plot(g, m, "-", color="C0", label="distribution mean (design)")
        ax.plot(g, best, ".", color="C2", ms=6, label="best design of the generation")
        ax.plot(gs, [lo[i] + span[i] * s["mean_geno"][i] for s in snaps], "x", color="C3", ms=7,
                label="internal mean, unclipped, checkpoints")
        ax.axhline(lo[i], ls="--", color="0.4", lw=1, label="search bounds")
        ax.axhline(lo[i] + span[i], ls="--", color="0.4", lw=1)
        ax.set_ylabel(f"{k}\nvalue", fontsize=8)
        axs[i, 1].semilogy(g, sd, "-o", ms=3, color="C1", label="sigma * sqrt(C_ii) in parameter units")
        axs[i, 1].set_ylabel(f"{k}\nstd ({sd[-1] / span[i]:.1%} of range last)", fontsize=8)
        log_ticks(axs[i, 1])
        for a in axs[i]:
            a.grid(alpha=0.3, which="both")
    axs[0, 0].set_title("value of the coordinate", fontsize=10)
    axs[0, 1].set_title("standard deviation of the coordinate", fontsize=10)
    axs[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.15), ncol=3, fontsize=8, borderaxespad=0.0)
    axs[0, 1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.15), fontsize=8, borderaxespad=0.0)
    for a in axs[-1]:
        a.set_xlabel("generation")
    fig.suptitle("Search distribution, best designs and bounds per coordinate", fontsize=11, y=1.03)
    fig.tight_layout()
    save(fig, out, "cmaes_coordinate.png")


def plot_principal_axes(snaps, gen, out):
    gs = [gen.get(s["iteration"], np.nan) for s in snaps]
    L = np.array([s["axis_len"] for s in snaps])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j in range(L.shape[1]):
        ax.semilogy(gs, L[:, j], "s-", label=f"axis {j + 1}" + (" (longest)" if j == 0 else " (shortest)" if j == L.shape[1] - 1 else ""))
    ax.set(xlabel="generation (checkpoints)", ylabel="axis length, sqrt eigenvalue of sampled covariance",
           title="Principal axis lengths of the search distribution")
    ax.grid(alpha=0.3, which="both")
    ax.legend(**LEGEND)
    save(fig, out, "cmaes_principal_axes.png")


def plot_principal_directions(snaps, gen, names, out):
    idx = evenly_spaced(len(snaps))
    fig, axs = plt.subplots(1, len(idx), figsize=(3.4 * len(idx) + 1.5, 4.2), squeeze=False,
                            constrained_layout=True)
    for ax, i in zip(axs[0], idx):
        s = snaps[i]
        labels = [f"{j + 1}: {l:.2g}" for j, l in enumerate(s["axis_len"])]
        im = heatmap(ax, s["axis_vec"] ** 2, labels, wrap(names), "viridis", 0, 1, ".2f",
                     f"generation {gen.get(s['iteration'], '?')}", ylabels=ax is axs[0, 0])
        ax.set_xlabel("axis, longest first, with length", fontsize=7)
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label="squared loading of parameter on axis")
    fig.suptitle("Composition of the principal axes", fontsize=10)
    save(fig, out, "cmaes_principal_directions.png")


def plot_covariance(snaps, gen, names, out):
    idx = evenly_spaced(len(snaps))
    fig, axs = plt.subplots(2, len(idx), figsize=(3.6 * len(idx) + 1.5, 7.8), squeeze=False,
                            constrained_layout=True)
    for col, i in enumerate(idx):
        s = snaps[i]
        v = np.abs(s["cov"]).max()
        im = heatmap(axs[0, col], s["cov"], wrap(names), wrap(names), "RdBu_r", -v, v, ".1e",
                     f"covariance, generation {gen.get(s['iteration'], '?')}", ylabels=col == 0)
        fig.colorbar(im, ax=axs[0, col], shrink=0.75, format="%.0e")
        imc = heatmap(axs[1, col], s["corr"], wrap(names), wrap(names), "RdBu_r", -1, 1, ".2f",
                      f"correlation, generation {gen.get(s['iteration'], '?')}", ylabels=col == 0)
    fig.colorbar(imc, ax=axs[1].tolist(), shrink=0.8, label="correlation coefficient")
    fig.suptitle("Sampled covariance (normalised units, own colour scale per panel) and correlation",
                 fontsize=10)
    save(fig, out, "cmaes_covariance.png")


def covariance_animation(snaps, gen, names, out, fps, show):
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    vmax = max(np.abs(s["cov"]).max() for s in snaps)
    gs = [gen.get(s["iteration"], np.nan) for s in snaps]
    L = np.array([s["axis_len"] for s in snaps])
    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.75)
    # Figure level axes, since clearing a heatmap each frame would delete an inset colour bar.
    cax = [fig.add_axes((ax.get_position().x1 + 0.008, ax.get_position().y0, 0.008, ax.get_position().height))
           for ax in axs[:2]]

    def frame(i):
        for ax in list(axs) + cax:
            ax.clear()
        s = snaps[i]
        v = np.abs(s["cov"]).max()
        im = heatmap(axs[0], s["cov"], wrap(names), wrap(names), "RdBu_r", -v, v, ".1e",
                     f"covariance (own scale, max |c| {v:.1e})")
        fig.colorbar(im, cax=cax[0], format="%.0e", label="covariance, normalised units")
        im = heatmap(axs[1], s["corr"], wrap(names), wrap(names), "RdBu_r", -1, 1, ".2f", "correlation", ylabels=False)
        fig.colorbar(im, cax=cax[1], label="correlation coefficient")
        for j in range(L.shape[1]):
            axs[2].semilogy(gs, L[:, j], "-", color=f"C{j}", alpha=0.35)
            axs[2].semilogy(gs[: i + 1], L[: i + 1, j], "o-", color=f"C{j}", ms=3, label=f"axis {j + 1}")
        axs[2].set(xlabel="generation", ylabel="principal axis length", title="axis lengths so far")
        axs[2].grid(alpha=0.3, which="both")
        axs[2].legend(**LEGEND)
        fig.suptitle(f"generation {gs[i]}, sigma {s['sigma']:.3g}, checkpoint {s['checkpoint']} "
                     f"({i + 1} of {len(snaps)}), global max |c| {vmax:.1e}", fontsize=10)

    anim = FuncAnimation(fig, frame, frames=len(snaps), interval=1000 / fps, repeat=True)
    gif = os.path.join(out, "cmaes_covariance_animation.gif")
    anim.save(gif, writer=PillowWriter(fps=fps), dpi=90)
    print("wrote", gif)
    try:
        import imageio_ffmpeg
        from matplotlib.animation import FFMpegWriter

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        mp4 = gif[:-4] + ".mp4"
        anim.save(mp4, writer=FFMpegWriter(fps=fps), dpi=110)
        print("wrote", mp4)
    except Exception as e:  # the gif is the portable record, the mp4 a convenience
        print("mp4 skipped:", e)
    if show:
        plt.show()
    plt.close(fig)


def plot_reliability(history, out):
    g = series(history, "generation")
    sh = series(history, "split_half_rank_correlation")
    rf = np.clip(2 * sh / (1 + sh), 0, 1)
    sd = series(history, "fitness_std")
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axs[0].plot(g, sh, "-o", ms=3, label="split-half rank correlation")
    axs[0].plot(g, rf, "--", label="full reliability 2r/(1+r), clipped at 0")
    axs[0].axhline(SPLIT_HALF_STOP, ls=":", color="k", label=f"stopping threshold {SPLIT_HALF_STOP}")
    axs[0].set(ylabel="correlation", ylim=(-1.05, 1.05), title="Reliability of the ranking")
    signal, noise = np.sqrt(rf) * sd, np.sqrt(1 - rf) * sd
    axs[1].plot(g, signal, "-o", ms=3, label="design signal std")
    axs[1].plot(g, noise, "-s", ms=3, label="evaluation noise std")
    robust_ylim(axs[1], [signal, noise])
    axs[1].set(xlabel="generation", ylabel="fitness std")
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
        ax.legend(**LEGEND)
    save(fig, out, "cmaes_reliability.png")


def plot_update_snr(history, names, out):
    g = series(history, "generation")
    snr = mean_shift_snr(history, names)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(g, snr, "-o", ms=3, label="SNR of the mean update, diagonal whitening")
    ax.axhline(0.0, color="k", lw=1)
    ax.axvspan(g[0], g[0] + 1 / SNR_BETA, color="0.85", label=f"estimator warm-up, 1/beta = {1 / SNR_BETA:.0f} generations")
    robust_ylim(ax, [snr[1:]])
    ax.set(xlabel="generation", ylabel="estimated SNR", title="Signal to noise ratio of the mean update")
    ax.grid(alpha=0.3)
    ax.legend(**LEGEND)
    save(fig, out, "cmaes_update_snr.png")


def plot_evaluation(history, out):
    g = series(history, "generation")
    keys = list(history[0])
    returns = sorted(k for k in keys if k.startswith("eval/return_row_"))
    surv = sorted(k for k in keys if k.startswith("eval/survival_row_"))
    fig, axs = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    axs[0].plot(g, series(history, "eval/survival"), "-o", ms=3, color="k", label="all rows")
    for k in surv:
        axs[0].plot(g, series(history, k), "-", lw=1, label=k.split("/")[-1].replace("survival_", ""))
    axs[0].set(ylabel="survival fraction", title="Evaluation outcome of the population")
    for k in returns:
        axs[1].plot(g, series(history, k), "-", lw=1, label=k.split("/")[-1].replace("return_", ""))
    robust_ylim(axs[1], [series(history, k) for k in returns], k=6.0)
    axs[1].set_ylabel("mean return per terrain row")
    axs[2].plot(g, series(history, "eval/lin_vel_error"), "-o", ms=3, color="C0", label="linear velocity error")
    axs[2].set_ylabel("linear velocity error, m/s")
    axs[3].plot(g, series(history, "eval/yaw_rate_error"), "-s", ms=3, color="C1", label="yaw rate error")
    axs[3].set(xlabel="generation", ylabel="yaw rate error, rad/s")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(**LEGEND)
    save(fig, out, "cmaes_evaluation.png")


def plot_population(snaps, gen, names, out, tol=0.01):
    snaps = [s for s in snaps if s["population"] is not None]
    if not snaps:
        return
    gs = [gen.get(s["iteration"], np.nan) + 1 for s in snaps]
    P = [np.clip(s["population"], 0, 1) for s in snaps]
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for i, k in enumerate(names):
        axs[0].plot(gs, [np.mean((p[:, i] < tol) | (p[:, i] > 1 - tol)) for p in P], "s-", label=k)
        axs[1].semilogy(gs, [p[:, i].std() for p in P], "s-", label=k)
    axs[0].set(ylabel=f"share within {tol:.0%} of a bound", ylim=(-0.03, 1.03),
               title="Population sampled at each checkpoint, boundary share and spread")
    axs[1].set(xlabel="generation of the sampled population", ylabel="std of designs, fraction of range")
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
        ax.legend(**LEGEND)
    save(fig, out, "cmaes_population.png")


def plot_categorical(history, snaps, gen, out):
    snaps = [s for s in snaps if s["q"] is not None]
    picks = [k for k in history[0] if k.startswith("pick_probability/")]
    if not snaps and not picks:
        return
    groups = snaps[0]["q"].shape[0] if snaps else 0
    fig, axs = plt.subplots(groups + 1, 1, figsize=(9, 2.8 * (groups + 1)), squeeze=False)
    axs = axs[:, 0]
    g = series(history, "generation")
    for k in picks:
        axs[0].plot(g, series(history, k), "-o", ms=3, label=k.split("/", 1)[1])
    axs[0].set(xlabel="generation", ylabel="largest category probability", ylim=(0, 1.02),
               title="Categorical search distribution")
    axs[0].grid(alpha=0.3)
    if picks:
        axs[0].legend(**LEGEND)
    gs = [gen.get(s["iteration"], np.nan) for s in snaps]
    for r in range(groups):
        K = int(snaps[0]["K"][r])
        Q = np.array([s["q"][r, :K] for s in snaps]).T
        im = axs[r + 1].imshow(Q, aspect="auto", cmap="magma", vmin=0, vmax=1, origin="lower",
                               extent=(-0.5, len(gs) - 0.5, -0.5, K - 0.5))
        axs[r + 1].set_xticks(range(len(gs)), [str(x) for x in gs], fontsize=7)
        axs[r + 1].set(xlabel="generation (checkpoints)", ylabel="category index",
                       title=f"categorical variable {r}, probability of each category")
        fig.colorbar(im, ax=axs[r + 1], label="probability")
    save(fig, out, "cmaes_categorical.png")


def write_summary(snaps, gen, names, lo, span, out):  # out is <exp_dir>/designs
    path = os.path.join(out, "cmaes_checkpoint_summary.csv")
    n = len(names)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "cma_iteration", "generation", "kind", "sigma", "path_ratio", "C_axis_ratio",
                    "sampled_axis_ratio"]
                   + [f"axis_len_{j + 1}" for j in range(n)]
                   + [f"mean_geno/{k}" for k in names] + [f"mean_internal_physical/{k}" for k in names]
                   + [f"scaling/{k}" for k in names]
                   + [f"corr/{names[i]}/{names[j]}" for i in range(n) for j in range(i + 1, n)])
        for s in snaps:
            w.writerow([s["checkpoint"], s["iteration"], gen.get(s["iteration"], ""), s["kind"], s["sigma"],
                        s["path_ratio"], s["C_axis_ratio"], s["axis_len"][0] / s["axis_len"][-1]]
                       + list(s["axis_len"]) + list(s["mean_geno"]) + list(lo + span * s["mean_geno"])
                       + list(s["scaling"]) + [s["corr"][i, j] for i in range(n) for j in range(i + 1, n)])
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dir")
    p.add_argument("--out", default=None, help="plot directory, default <exp_dir>/designs/plots")
    p.add_argument("--show", action="store_true", help="open the covariance animation in a window")
    p.add_argument("--fps", type=float, default=2.0, help="frames per second of the animation")
    a = p.parse_args()

    global plt, mticker
    import matplotlib

    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    designs = os.path.join(a.exp_dir, "designs")
    out = a.out or os.path.join(designs, "plots")
    os.makedirs(out, exist_ok=True)
    run = os.path.join(*os.path.normpath(a.exp_dir).split(os.sep)[-2:])
    snaps, history = load_run(a.exp_dir)
    gen = generation_of(history)
    names = param_names(history)
    lo, span = infer_ranges(snaps, history, names)
    print(f"{run}, {snaps[0]['kind']}, {len(history)} generations, {len(snaps)} checkpoints, "
          "ranges " + ", ".join(f"{k} [{l:.4g}, {l + s:.4g}]" for k, l, s in zip(names, lo, span)))

    plot_fitness(history, out)
    plot_step_size(history, snaps, gen, out)
    plot_coordinate(history, snaps, gen, names, lo, span, out)
    plot_principal_axes(snaps, gen, out)
    plot_principal_directions(snaps, gen, names, out)
    plot_covariance(snaps, gen, names, out)
    plot_reliability(history, out)
    plot_update_snr(history, names, out)
    plot_evaluation(history, out)
    plot_population(snaps, gen, names, out)
    plot_categorical(history, snaps, gen, out)
    write_summary(snaps, gen, names, lo, span, designs)
    covariance_animation(snaps, gen, names, out, a.fps, a.show)


if __name__ == "__main__":
    main()
