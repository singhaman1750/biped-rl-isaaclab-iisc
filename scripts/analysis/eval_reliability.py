"""Reliability of the controlled CMA-ES evaluation against horizon and column count.

Reads designs/eval_trace_<g>.pt written with eval_save_trace and prints, per horizon and
number of columns, the split-half rank correlation over even and odd columns, the full
reliability and the two variance components of section 3.4 of plans/copt_improvement.md.

Usage:
    python scripts/analysis/eval_reliability.py <eval_trace.pt>
"""

import argparse

import torch


def ranks(x):
    return x.argsort().argsort().float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    a = p.parse_args()
    d = torch.load(a.trace)
    n, dt = d["num_designs"], d["step_dt"]
    reward = d["reward"].float()                              # (R, P, T, m * n)
    rows, passes, steps_max = reward.shape[:3]
    reward = reward.view(rows, passes, steps_max, -1, n)       # (R, P, T, m, n), env = j * n + k
    col = d["column"]                                          # (P, m)
    pair = col // 2
    print("horizon_s columns split_half rho_full var_design var_noise_half")
    for h in (1, 2, 4, 8, 16):
        steps = min(int(round(h / dt)), steps_max)
        ret = reward[:, :, :steps].sum(2)                      # (R, P, m, n)
        for k in (4, 2, 1):
            keep = pair % k == 0                               # whole column pairs, both halves kept
            even, odd = keep & (col % 2 == 0), keep & (col % 2 == 1)
            fa, fb = ret[:, even].mean((0, 1)), ret[:, odd].mean((0, 1))
            sh = torch.corrcoef(torch.stack([ranks(fa), ranks(fb)]))[0, 1].item()
            full = 2 * sh / (1 + sh) if sh > -1 else float("nan")
            var_d = torch.cov(torch.stack([fa, fb]))[0, 1].item()
            var_h = 0.5 * (fa - fb).var().item()
            print(f"{h:9.0f} {col[keep].unique().numel():7d} {sh:10.3f} {full:8.3f} {var_d:10.4f} {var_h:14.4f}")
        if steps == steps_max:
            break


if __name__ == "__main__":
    main()
