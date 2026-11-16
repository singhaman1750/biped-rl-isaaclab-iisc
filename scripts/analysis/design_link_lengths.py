"""Link length statistics of the co-optimisation design dumps.

Reads the per generation design dumps that the COPT runner writes into a run's designs/
folder and plots the mean, the median and the standard deviation of the thigh and shank
lengths over the population against the learning iteration at which each dump was taken.

Two formats are read. The legacy link_lengths_{it}.csv has no header, an index column and
one stringified dict per body, the thigh entry recognised by its knee_*_Joint child and the
shank entry by its ankle_*_act child, and the length is recovered from the box centre, which
_biped_links places at -l/2 on the link axis. The current design_parameters_{it}.csv carries
thigh_length and shank_length as plain columns.

Usage:
    python scripts/analysis/design_link_lengths.py <designs_dir> [--side R] [--out PATH]

Pure numpy, pandas and matplotlib, so it runs in a plain interpreter against a stored run.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LINKS = ("thigh", "shank")
COLOURS = {"thigh": "#2a78d6", "shank": "#eb6834"}
STATISTICS = ("mean", "median", "std")


def _read_legacy(path: str, side: str) -> dict[str, np.ndarray]:
    csv.field_size_limit(1 << 30)
    lengths = {link: [] for link in LINKS}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            for cell in row[1:]:
                if "child_joint" not in cell:
                    continue
                entry = ast.literal_eval(cell)
                if entry["child_joint"] == f"knee_{side}_Joint":
                    lengths["thigh"].append(-2.0 * entry["box_pos"][2])
                elif entry["child_joint"] == f"ankle_{side}_act":
                    lengths["shank"].append(-2.0 * entry["box_pos"][2])
    return {link: np.asarray(v) for link, v in lengths.items()}


def _read_parameters(path: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    return {link: df[f"{link}_length"].to_numpy(dtype=float) for link in LINKS}


def read_designs(path: str, side: str = "R") -> dict[str, np.ndarray]:
    """Thigh and shank lengths in metres of every individual in one dump."""
    if os.path.basename(path).startswith("design_parameters_"):
        return _read_parameters(path)
    return _read_legacy(path, side)


def summarise(designs_dir: str, side: str = "R") -> pd.DataFrame:
    """One row per dump, the statistics of each link over the population."""
    paths = glob.glob(os.path.join(designs_dir, "link_lengths_*.csv"))
    paths += glob.glob(os.path.join(designs_dir, "design_parameters_*.csv"))
    rows = []
    for path in paths:
        iteration = int(re.search(r"_(\d+)\.csv$", path).group(1))
        lengths = read_designs(path, side)
        row: dict[str, float] = {"iteration": iteration, "individuals": len(lengths["thigh"])}
        for link, v in lengths.items():
            row[f"{link}_mean"] = v.mean()
            row[f"{link}_median"] = np.median(v)
            row[f"{link}_std"] = v.std(ddof=1) if len(v) > 1 else 0.0
            row[f"{link}_min"] = v.min()
            row[f"{link}_max"] = v.max()
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no design dumps in {designs_dir}")
    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


def plot(summary: pd.DataFrame, out: str, title: str = "") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, stat in zip(axes, STATISTICS):
        for link in LINKS:
            ax.plot(summary["iteration"], summary[f"{link}_{stat}"], color=COLOURS[link],
                    linewidth=2, marker="o", markersize=4, label=link)
            ax.annotate(f"{summary[f'{link}_{stat}'].iloc[-1]:.4f}",
                        (summary["iteration"].iloc[-1], summary[f"{link}_{stat}"].iloc[-1]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8, color="#52514e")
        for it in summary["iteration"][summary["individuals"].diff().fillna(0) != 0]:
            ax.axvline(it, color="#a3a29c", linewidth=1, linestyle="--")
            ax.annotate("population size change", (it, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points", fontsize=8, color="#52514e")
        ax.set_title(f"{stat} link length", fontsize=11)
        ax.set_xlabel("learning iteration")
        ax.set_ylabel("length (m)")
        ax.grid(True, color="#e5e4e0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="center right")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("designs_dir")
    parser.add_argument("--side", default="R", choices=("R", "L"))
    parser.add_argument("--out", default=None, help="defaults to <designs_dir>/link_length_statistics.png")
    args = parser.parse_args()

    summary = summarise(args.designs_dir, args.side)
    out = args.out or os.path.join(args.designs_dir, "link_length_statistics.png")
    run = os.path.basename(os.path.dirname(os.path.abspath(args.designs_dir)))
    plot(summary, out, title=f"Population link lengths, {run}")
    summary.to_csv(os.path.splitext(out)[0] + ".csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
