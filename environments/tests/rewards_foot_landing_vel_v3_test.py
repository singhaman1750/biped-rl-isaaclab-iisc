"""Numerical validation of ``foot_landing_vel_v3``.

The function is extracted from the source of ``rewards.py`` by ``ast`` rather than imported,
for the reason ``rewards_foot_clearance_v4_test.py`` gives, that importing the module would
pull in Isaac Lab and that extracting from source keeps the suite from drifting away from the
shipped implementation. The Isaac Lab objects the function reaches through are replaced by
plain stubs exposing only the attributes it touches, and ``quat_apply`` by a direct
implementation of the Hamilton rotation, so that the arithmetic under test is the real one.

The cases exercise the point foot branch and the sole foot branch separately, and establish
in both directions that the terrain reference changes the verdict where an absolute height
gate would get it wrong, missing an imminent strike on a raised tread and firing spuriously
over a pit.

Run with ``python3 -m pytest environments/tests/rewards_foot_landing_vel_v3_test.py``, or
directly with ``python3 environments/tests/rewards_foot_landing_vel_v3_test.py``.
"""
from __future__ import annotations

import ast
import pathlib

import torch

_REWARDS = pathlib.Path(__file__).resolve().parents[1] / "environments/tasks/locomotion/mdp/rewards.py"
_WANTED = ("_terrain_height_under_points", "_sole_points_world", "foot_landing_vel_v3")


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the Hamilton quaternion q, both flattened to (M, 4) and (M, 3)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


class _MathUtils:
    quat_apply = staticmethod(_quat_apply)


def _load():
    src = _REWARDS.read_text()
    lines = src.split("\n")
    ns: dict = {
        "torch": torch,
        "math_utils": _MathUtils,
        # names appearing in evaluated signature annotations only
        "ManagerBasedRLEnv": object,
        "SceneEntityCfg": object,
        "Articulation": object,
        "RayCaster": object,
        "ContactSensor": object,
    }
    found = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED:
            exec("\n".join(lines[node.lineno - 1 : node.end_lineno]), ns)
            found[node.name] = ns[node.name]
    missing = set(_WANTED) - set(found)
    assert not missing, f"not found in {_REWARDS}: {sorted(missing)}"
    return found["foot_landing_vel_v3"]


landing_vel_v3 = _load()

RES, SX, SY = 0.1, 1.6, 1.0
FOOT_RADIUS = 0.022


class _Cfg:
    def __init__(self, name, body_ids):
        self.name, self.body_ids = name, body_ids


class _Data:
    pass


class _Obj:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, robot, sensors):
        self._robot, self.sensors = robot, sensors

    def __getitem__(self, key):
        return self._robot


class _Env:
    def __init__(self, scene):
        self.scene = scene


def _grid(height_fn):
    """One environment's ray hits over the real scanner footprint, heights from height_fn."""
    xs = torch.arange(-SX / 2, SX / 2 + 1e-9, RES)
    ys = torch.arange(-SY / 2, SY / 2 + 1e-9, RES)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    return torch.stack([gx, gy, height_fn(gx, gy)], dim=-1).unsqueeze(0)


def _build(foot_xyz, lin_vel, ang_vel, contact_force, height_fn, quat=None):
    """Assemble a one environment, F foot scene. foot_xyz is (F, 3)."""
    f = foot_xyz.shape[0]
    d = _Data()
    d.body_link_pos_w = foot_xyz.unsqueeze(0)
    d.body_link_lin_vel_w = lin_vel.unsqueeze(0)
    d.body_link_ang_vel_w = ang_vel.unsqueeze(0)
    d.body_pos_w = d.body_link_pos_w
    d.body_quat_w = (quat if quat is not None else torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(f, 1)).unsqueeze(0)
    robot = _Obj(d)

    hd = _Data()
    hd.ray_hits_w = _grid(height_fn)
    scan = _Obj(hd)

    cd = _Data()
    # (N, history, bodies, 3)
    cd.net_forces_w_history = torch.zeros(1, 4, f, 3)
    cd.net_forces_w_history[0, :, :, 2] = contact_force
    contact = _Obj(cd)

    env = _Env(_Scene(robot, {"height_scanner": scan, "contact_forces": contact}))
    return env, _Cfg("robot", slice(None)), _Cfg("contact_forces", slice(None)), _Cfg("height_scanner", slice(None))


FLAT = lambda x, y: torch.zeros_like(x)
STEP_08 = lambda x, y: torch.where(x > 0.0, torch.full_like(x, 0.08), torch.zeros_like(x))
PIT_08 = lambda x, y: torch.where(x > 0.0, torch.full_like(x, -0.08), torch.zeros_like(x))

RESULTS: list[tuple[str, bool, str]] = []


def _check(name, got, want, tol=1e-6):
    ok = abs(float(got) - want) <= tol
    RESULTS.append((name, ok, f"got {float(got):.6f} want {want:.6f}"))
    assert ok, f"{name}: got {float(got)} want {want}"


def test_point_foot_flat_charges_the_approach():
    """Contact point 28 mm above flat ground, descending at 1.5 m/s, inside a 40 mm gate."""
    env, a, s, h = _build(
        torch.tensor([[0.0, 0.0, 0.05]]), torch.tensor([[0.0, 0.0, -1.5]]),
        torch.zeros(1, 3), 0.0, FLAT,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, foot_radius=FOOT_RADIUS)
    _check("point foot, flat ground, inside gate", r[0], 1.5**2)


def test_point_foot_over_a_step_is_charged_where_an_absolute_gate_would_miss_it():
    """Foot 78 mm above the datum but 2 mm BELOW the 80 mm tread it is about to strike."""
    env, a, s, h = _build(
        torch.tensor([[0.3, 0.0, 0.10]]), torch.tensor([[0.0, 0.0, -1.2]]),
        torch.zeros(1, 3), 0.0, STEP_08,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, foot_radius=FOOT_RADIUS)
    # an absolute gate sees 0.10 - 0.022 = 0.078 > 0.04 and stays shut
    _check("point foot, raised tread, gate opens", r[0], 1.2**2)


def test_point_foot_over_a_pit_is_not_charged_where_an_absolute_gate_would_fire():
    """Foot 28 mm above the datum but 108 mm above the floor of the pit beneath it."""
    env, a, s, h = _build(
        torch.tensor([[0.3, 0.0, 0.05]]), torch.tensor([[0.0, 0.0, -1.2]]),
        torch.zeros(1, 3), 0.0, PIT_08,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, foot_radius=FOOT_RADIUS)
    # an absolute gate sees 0.028 < 0.04 and fires
    _check("point foot, pit, gate stays shut", r[0], 0.0)


def test_a_loaded_foot_is_exempt():
    env, a, s, h = _build(
        torch.tensor([[0.0, 0.0, 0.05]]), torch.tensor([[0.0, 0.0, -1.5]]),
        torch.zeros(1, 3), 50.0, FLAT,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, foot_radius=FOOT_RADIUS,
                       force_threshold=1.0)
    _check("loaded foot exempt", r[0], 0.0)


def test_a_rising_foot_is_exempt():
    env, a, s, h = _build(
        torch.tensor([[0.0, 0.0, 0.05]]), torch.tensor([[0.0, 0.0, 1.5]]),
        torch.zeros(1, 3), 0.0, FLAT,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, foot_radius=FOOT_RADIUS)
    _check("rising foot exempt", r[0], 0.0)


def test_a_foot_outside_the_scanner_is_not_charged():
    env, a, s, h = _build(
        torch.tensor([[5.0, 5.0, 0.05]]), torch.tensor([[0.0, 0.0, -1.5]]),
        torch.zeros(1, 3), 0.0, FLAT,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, foot_radius=FOOT_RADIUS,
                       max_horizontal_dist=0.5)
    _check("unresolvable foot not charged", r[0], 0.0)


def test_sole_foot_charges_the_rotational_term():
    """Level sole, no frame velocity, pitching at 1 rad/s about y with the low point at x=0.1.

    The vertical component of ``omega x r`` is ``omega_x r_y - omega_y r_x``, here
    ``0 - 1.0 * 0.1 = -0.1``, so the contact point descends at 0.1 m/s although the frame
    itself does not move at all. A term reading only the frame velocity charges nothing.
    """
    offsets = [[0.10, 0.0, -0.02], [-0.10, 0.0, -0.01]]
    env, a, s, h = _build(
        torch.tensor([[0.0, 0.0, 0.05]]), torch.zeros(1, 3),
        torch.tensor([[0.0, 1.0, 0.0]]), 0.0, FLAT,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, sole_offsets=offsets)
    # lowest point at z = 0.05 - 0.02 = 0.03 above flat ground, inside the 0.04 gate
    _check("sole foot, rotational approach", r[0], 0.1**2)


def test_sole_foot_uses_the_lowest_point_for_the_gate():
    """Raising the foot so the lowest sole point clears the gate silences the term."""
    offsets = [[0.10, 0.0, -0.02], [-0.10, 0.0, -0.01]]
    env, a, s, h = _build(
        torch.tensor([[0.0, 0.0, 0.10]]), torch.zeros(1, 3),
        torch.tensor([[0.0, 1.0, 0.0]]), 0.0, FLAT,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, sole_offsets=offsets)
    # lowest point at 0.08 above ground, outside the gate
    _check("sole foot, above the gate", r[0], 0.0)


def test_four_feet_sum():
    """A quadruped's four feet, two of them inside the gate, the penalty summing over them."""
    env, a, s, h = _build(
        torch.tensor([[0.2, -0.14, 0.05], [0.2, 0.14, 0.30], [-0.2, -0.14, 0.05], [-0.2, 0.14, 0.30]]),
        torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -3.0], [0.0, 0.0, -2.0], [0.0, 0.0, -3.0]]),
        torch.zeros(4, 3), 0.0, FLAT,
    )
    r = landing_vel_v3(env, a, s, h, about_landing_threshold=0.04, foot_radius=FOOT_RADIUS)
    _check("four feet, two inside the gate", r[0], 1.0**2 + 2.0**2)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    w = max(len(n) for n, _, _ in RESULTS)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:{w}s}  {detail}")
    print(f"\n{sum(1 for _, ok, _ in RESULTS if ok)}/{len(RESULTS)} checks passed")
