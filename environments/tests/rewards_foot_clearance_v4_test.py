"""Numerical validation of the terrain lookup behind ``foot_clearance_reward_v4``.

The reward itself cannot be exercised without a simulator, but its numerical core can, which
is why ``_terrain_height_under_points`` was factored out of it as a function over plain
tensors carrying no Isaac Lab dependency. That function is extracted here from the source of
``rewards.py`` by ``ast`` rather than imported, because importing the module would pull in
Isaac Lab, and extracting from source has the further merit that the suite cannot drift away
from the shipped implementation, it is always testing the file as it stands.

The synthetic terrain reproduces the real scanner geometry, a GridPatternCfg of resolution
0.1 m over a size of (1.6, 1.0), giving 17 by 11 = 187 rays, and the quadruped's nominal foot
positions at plus and minus 0.20188 m fore and aft by plus and minus 0.14200 m laterally.

Run with ``python3 -m pytest environments/tests/rewards_foot_clearance_v4_test.py``, or
directly with ``python3 environments/tests/rewards_foot_clearance_v4_test.py`` for a report.
"""
from __future__ import annotations

import ast
import pathlib

import torch

_REWARDS = pathlib.Path(__file__).resolve().parents[1] / "environments/tasks/locomotion/mdp/rewards.py"


def _load_helper():
    """Extract ``_terrain_height_under_points`` from the live rewards module by source."""
    src = _REWARDS.read_text()
    lines = src.split("\n")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_terrain_height_under_points":
            ns: dict = {"torch": torch}
            exec("\n".join(lines[node.lineno - 1 : node.end_lineno]), ns)
            return ns["_terrain_height_under_points"]
    raise AssertionError(f"_terrain_height_under_points not found in {_REWARDS}")


lookup = _load_helper()

RES, SX, SY = 0.1, 1.6, 1.0
FEET = torch.tensor([[0.20188, -0.14200], [0.20188, 0.14200], [-0.20188, -0.14200], [-0.20188, 0.14200]])
SLOPE = 0.3
STEP = 0.08  # the tallest riser the terrain generator produces


def _grid() -> torch.Tensor:
    x = torch.arange(-SX / 2, SX / 2 + 1e-9, RES)
    y = torch.arange(-SY / 2, SY / 2 + 1e-9, RES)
    gx, gy = torch.meshgrid(x, y, indexing="xy")
    return torch.stack([gx.flatten(), gy.flatten()], -1)


def _hits(surface) -> torch.Tensor:
    xy = _grid()
    return torch.cat([xy, surface(xy[:, 0]).unsqueeze(-1)], -1).unsqueeze(0)


def _feet(z: float = 0.15) -> torch.Tensor:
    return torch.cat([FEET, torch.full((4, 1), z)], -1).unsqueeze(0)


_flat = lambda x: torch.zeros_like(x)
_slope = lambda x: SLOPE * x
_step = lambda x: torch.where(x > 0, torch.full_like(x, STEP), torch.zeros_like(x))


def test_grid_matches_the_configured_scanner():
    assert _grid().shape[0] == 187


def test_flat_terrain_is_exact():
    z, valid = lookup(_feet(), _hits(_flat))
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)
    assert valid.all()


def test_offset_flat_terrain_gives_the_true_clearance():
    z, _ = lookup(_feet(z=0.52), _hits(lambda x: torch.full_like(x, 0.37)))
    assert torch.allclose(0.52 - z, torch.full_like(z, 0.15), atol=1e-6)


def test_per_foot_lookup_beats_the_global_mean_on_a_slope():
    """The measurement that justifies the whole construction."""
    hits = _hits(_slope)
    truth = _slope(FEET[:, 0]).unsqueeze(0)
    z, _ = lookup(_feet(), hits, reduction="mean")
    global_mean_err = (hits[..., 2].mean() - truth).abs().max()
    assert (z - truth).abs().max() < 2e-3
    # the mean over all rays, which base_height_rough_l2 takes, is wrong for every foot
    assert global_mean_err > 0.05


def test_max_reduction_bias_on_a_slope_is_half_a_cell_diagonal():
    z, _ = lookup(_feet(), _hits(_slope), reduction="max")
    bias = (z - _slope(FEET[:, 0]).unsqueeze(0)).mean()
    assert abs(bias - 0.03) < 0.01


def test_feet_straddling_a_step_resolve_to_different_treads():
    z, _ = lookup(_feet(), _hits(_step))
    assert torch.allclose(z[0, :2], torch.full((2,), STEP), atol=1e-6)  # front, upper tread
    assert torch.allclose(z[0, 2:], torch.zeros(2), atol=1e-6)  # rear, lower tread


def test_max_reduction_does_not_announce_a_foot_clear_at_a_riser():
    """The reason ``max`` is the default. A mean here reports more than double the clearance."""
    pts = torch.tensor([[[0.02, 0.0, 0.15]]])
    hits = _hits(_step)
    z_mean, _ = lookup(pts, hits, reduction="mean")
    z_max, _ = lookup(pts, hits, reduction="max")
    assert (0.15 - z_max).item() < 0.075  # true clearance is 0.070
    assert (0.15 - z_mean).item() > 0.14  # the mean claims the foot is clear


def test_four_neighbours_is_the_measured_optimum():
    hits = _hits(_slope)
    xs = torch.linspace(-0.10, 0.10, 41)
    pts = torch.stack([xs, torch.full_like(xs, 0.031), torch.full_like(xs, 0.15)], -1).unsqueeze(0)
    err = {}
    for k in (1, 3, 4, 6, 9):
        z, _ = lookup(pts, hits, num_neighbours=k, reduction="mean")
        err[k] = (z[0] - _slope(xs)).abs().max().item()
    assert err[4] < 2e-3
    assert err[4] < err[1] and err[4] < err[6] and err[4] < err[9]


def test_dead_rays_are_excluded_from_selection_not_substituted():
    hits = _hits(_slope).clone()
    nearest = torch.topk(((hits[0, :, :2] - FEET[0]) ** 2).sum(-1), 2, largest=False).indices
    hits[0, nearest, 2] = torch.inf
    z, valid = lookup(_feet(), hits)
    assert torch.isfinite(z).all()
    assert valid.all()


def test_a_foot_with_no_live_ray_is_reported_invalid():
    hits = _hits(_flat).clone()
    hits[..., 2] = torch.inf
    z, valid = lookup(_feet(), hits)
    assert not valid.any()
    assert torch.isfinite(z).all()


def test_a_foot_outside_the_footprint_is_rejected():
    far = torch.tensor([[[5.0, 0.0, 0.15]]])
    _, limited = lookup(far, _hits(_flat), max_horizontal_dist=0.5)
    _, unlimited = lookup(far, _hits(_flat), max_horizontal_dist=None)
    assert not limited.item()
    assert unlimited.item()


def test_shapes_and_batching():
    n = 64
    z, valid = lookup(_feet().repeat(n, 1, 1), _hits(_slope).repeat(n, 1, 1))
    assert z.shape == (n, 4) and valid.shape == (n, 4)
    assert torch.allclose(z, z[:1].expand(n, 4))


def test_an_unknown_reduction_is_rejected():
    try:
        lookup(_feet(), _hits(_flat), reduction="median")
    except ValueError:
        return
    raise AssertionError("an unknown reduction must raise ValueError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
