"""Numerical validation of ``feet_hold_penalty``, phase 2 of the degenerate gait plan.

The function is extracted from the source of ``rewards.py`` by ``ast`` rather than imported,
because importing the module would pull in Isaac Lab, and extracting from source has the
further merit that the suite cannot drift away from the shipped implementation, it is always
testing the file as it stands. The only Isaac Lab surfaces the function touches are a contact
sensor's ``current_air_time`` and the command manager, both of which are stubbed here.

The reference figures are taken from the evaluation dumps the term was derived against,
``2026-09-08_07-37-52`` for a healthy four beat gait and ``2026-09-09_11-10-10`` for the rear
left retraction, so the suite fails if the arithmetic that produced the configured weight of
-0.1 ever ceases to hold.

Run with ``python3 -m pytest environments/tests/rewards_feet_hold_penalty_test.py``, or
directly with ``python3 environments/tests/rewards_feet_hold_penalty_test.py`` for a report.
"""
from __future__ import annotations

import ast
import pathlib

import torch

_REWARDS = pathlib.Path(__file__).resolve().parents[1] / "environments/tasks/locomotion/mdp/rewards.py"

CEILING = 0.45          # air_time_ceiling configured for the quadruped
WEIGHT = -0.1           # weight configured for the quadruped
EPISODE_S = 20.0        # episode_length_s
DT = 0.02               # step_dt


def _load_term():
    """Extract ``feet_hold_penalty`` from the live rewards module by source."""
    src = _REWARDS.read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "feet_hold_penalty"
    )
    fn.decorator_list = []
    ns = {"torch": torch}
    # The annotations reference Isaac Lab types that are not importable here. They are only
    # annotations, so compiling with `from __future__ import annotations` semantics (PEP 563)
    # leaves them as strings and never evaluates them.
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, "<rewards.py:feet_hold_penalty>", "exec", flags=0x1000000)  # annotations
    exec(code, ns)
    return ns["feet_hold_penalty"]


feet_hold_penalty = _load_term()


class _Sensor:
    def __init__(self, air_time):
        self.data = type("D", (), {"current_air_time": air_time})()


class _CommandManager:
    def __init__(self, command):
        self._command = command

    def get_command(self, name):
        return self._command


class _Env:
    """Minimal stand-in exposing only the two surfaces the term reads."""

    def __init__(self, air_time, command_speed):
        n = air_time.shape[0]
        cmd = torch.zeros(n, 3)
        cmd[:, 0] = command_speed
        self.scene = type("S", (), {"sensors": {"contact_forces": _Sensor(air_time)}})()
        self.command_manager = _CommandManager(cmd)


class _Cfg:
    name = "contact_forces"
    body_ids = slice(None)


def _call(air_time, command_speed=1.0, ceiling=CEILING):
    env = _Env(air_time, command_speed)
    return feet_hold_penalty(env, _Cfg(), "base_velocity", ceiling)


# --------------------------------------------------------------------------------------
# Behaviour under the ceiling
# --------------------------------------------------------------------------------------

def test_all_feet_below_ceiling_pay_exactly_zero():
    """A gait whose swings complete within the ceiling is charged nothing at all."""
    air = torch.tensor([[0.00, 0.14, 0.30, 0.449]])
    assert torch.allclose(_call(air), torch.zeros(1)), "a healthy gait must be free"


def test_measured_swing_distribution_pays_zero():
    """The measured air times of both runs are near 0.14 s, far under the 0.45 s ceiling."""
    torch.manual_seed(0)
    air = torch.clamp(torch.randn(256, 4) * 0.03 + 0.14, min=0.0)
    assert float(_call(air).max()) == 0.0, "the measured swing distribution must be free"


def test_charge_begins_exactly_at_the_ceiling():
    """The term is continuous at the ceiling and has unit slope above it."""
    assert float(_call(torch.tensor([[CEILING, 0.0, 0.0, 0.0]]))) == 0.0
    got = float(_call(torch.tensor([[CEILING + 0.10, 0.0, 0.0, 0.0]])))
    assert abs(got - 0.10) < 1e-6, f"slope above the ceiling must be 1, got {got}"


# --------------------------------------------------------------------------------------
# Behaviour above the ceiling, the property the term exists for
# --------------------------------------------------------------------------------------

def test_unclipped_growth_is_unbounded_and_linear():
    """No clip: the per step charge keeps growing, which is the whole point of the term.

    Spot's GaitReward clamps its errors at ``max_err**2``, so a diagonal pair held aloft
    together scores near its ceiling. This term must not acquire that defect.
    """
    held = [1.0, 5.0, 10.0, 20.0, 100.0]
    got = [float(_call(torch.tensor([[h, 0.0, 0.0, 0.0]]))) for h in held]
    assert got == sorted(got), "charge must be monotone in the hold duration"
    for h, g in zip(held, got):
        assert abs(g - (h - CEILING)) < 1e-4, f"charge at {h}s must be {h - CEILING}, got {g}"
    assert got[-1] > 99.0, "a 100 s hold must not saturate"


def test_episode_integral_is_quadratic_in_the_hold():
    """Integrating the linear per step charge over an episode gives the (T-c)^2/2 law.

    This is the derivation behind the configured weight, so it is asserted rather than
    trusted: one limb held for a whole episode accrues (20 - 0.45)^2 / 2 = 191.10 s.s,
    which at -0.1 is -19.11 reward units.
    """
    steps = int(EPISODE_S / DT)
    total = 0.0
    for i in range(steps):
        air = torch.tensor([[i * DT, 0.0, 0.0, 0.0]])
        total += float(_call(air)) * DT
    closed_form = (EPISODE_S - CEILING) ** 2 / 2.0
    assert abs(total - closed_form) < 0.5, f"integral {total:.2f} vs closed form {closed_form:.2f}"
    assert abs(total * WEIGHT - (-19.11)) < 0.06, "the -19.11 figure in the config must hold"


def test_a_hold_twice_as_long_costs_four_times_as_much():
    """The quadratic law, stated as the property that makes a runaway inescapable."""
    def integral(hold_s):
        n = int(hold_s / DT)
        return sum(float(_call(torch.tensor([[i * DT, 0.0, 0.0, 0.0]]))) * DT for i in range(n))
    short, long = integral(5.0), integral(10.0)
    ratio = long / short
    assert 3.5 < ratio < 4.5, f"doubling the hold must roughly quadruple the cost, got {ratio:.2f}"


def test_two_held_limbs_cost_twice_one():
    """The sum over feet is linear, so the two legged hop is charged twice the three legged."""
    one = float(_call(torch.tensor([[8.0, 0.1, 0.1, 0.1]])))
    two = float(_call(torch.tensor([[8.0, 8.0, 0.1, 0.1]])))
    assert abs(two - 2.0 * one) < 1e-4, f"two limbs must cost twice one, got {two} against {one}"


# --------------------------------------------------------------------------------------
# The command gate
# --------------------------------------------------------------------------------------

def test_zero_command_is_not_charged():
    """A robot commanded to stand is not charged for holding still."""
    air = torch.tensor([[20.0, 20.0, 20.0, 20.0]])
    assert float(_call(air, command_speed=0.0)) == 0.0


def test_gate_threshold_matches_feet_air_time_v2():
    """The default gate is 0.1 m/s, the figure feet_air_time_v2 hard-codes."""
    air = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    assert float(_call(air, command_speed=0.09)) == 0.0, "below 0.1 m/s must be free"
    assert float(_call(air, command_speed=0.11)) > 0.0, "above 0.1 m/s must charge"


# --------------------------------------------------------------------------------------
# Separation against the failure the term was written for
# --------------------------------------------------------------------------------------

def test_separates_the_measured_failure_from_the_measured_healthy_gait():
    """Run B pays nothing and run C pays enough to matter, which is the design claim.

    Run B (2026-09-08_07-37-52) has all four feet load bearing in all 32 environments with
    air times near 0.14 s. Run C (2026-09-09_11-10-10) holds the rear left foot. The two
    must be separated by far more than the 1.86 reward units by which their total budgets
    actually differ, or the term cannot steer the policy.
    """
    torch.manual_seed(1)
    healthy = torch.clamp(torch.randn(32, 4) * 0.03 + 0.14, min=0.0)
    retracted = healthy.clone()
    retracted[:, 2] = torch.linspace(0.5, 19.5, 32)      # rear left, held and accruing

    b = float(_call(healthy).mean()) * WEIGHT * EPISODE_S
    c = float(_call(retracted).mean()) * WEIGHT * EPISODE_S
    assert abs(b) < 1e-6, f"the healthy gait must be free, charged {b}"
    assert c < -15.0, f"the retraction must be charged decisively, charged {c}"
    assert abs(c - b) > 8.0 * 1.86, "separation must exceed the observed budget indifference"


def test_per_foot_attribution_is_recoverable():
    """The sum is over feet, so a single held foot's charge is attributable to it alone."""
    air = torch.tensor([[0.1, 0.1, 12.0, 0.1]])
    total = float(_call(air))
    assert abs(total - (12.0 - CEILING)) < 1e-5, "only the held foot may contribute"


# --------------------------------------------------------------------------------------
# Shapes and batching
# --------------------------------------------------------------------------------------

def test_shape_is_per_environment():
    air = torch.rand(7, 4) * 2.0
    assert _call(air).shape == (7,)


def test_sign_is_positive_so_the_config_supplies_the_negative_weight():
    """The function returns a positive excess; the configuration makes it a penalty."""
    air = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    assert float(_call(air)) > 0.0


def test_body_ids_subset_is_respected():
    """Only the resolved feet contribute, not every body the sensor carries."""
    class _Subset:
        name = "contact_forces"
        body_ids = [0, 2]
    env = _Env(torch.tensor([[9.0, 9.0, 9.0, 9.0]]), 1.0)
    got = float(feet_hold_penalty(env, _Subset(), "base_velocity", CEILING))
    assert abs(got - 2 * (9.0 - CEILING)) < 1e-5, f"only two feet must contribute, got {got}"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as error:
            failed += 1
            print(f"  FAIL  {name}: {error}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
