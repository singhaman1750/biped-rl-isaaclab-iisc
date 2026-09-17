# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-test for the kscale sagittal symmetry mirror, without Isaac Lab or a running simulator.

The mirror is an order-2 reflection, so mirroring twice must be the identity, and the joint and
body permutations must each be involutions. A mirror that is silently wrong does not raise, it
merely trains a policy on a false invariance, so these properties are worth asserting before a
launch rather than inferring from a reward curve that fails to rise.

The observation manager and the articulation are stubbed with the term layout and the runtime
name lists that the kscale configuration produces, so the test exercises the real permutation
building and the real block dispatch. It does not exercise Isaac Sim's actual name ordering,
which is why the module builds its maps from the runtime names rather than assuming an order.

Usage:
    python kscale_symmetry_selftest.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

# Loaded by file path rather than by package import. The `environments` package __init__ chain
# pulls in Isaac Lab, toml and the whole task registry, none of which the mirror itself needs,
# and requiring them here would make the test runnable only inside the simulation container.
_MODULE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "environments", "environments", "tasks", "locomotion", "mdp", "symmetry",
    "kscale.py",
)
_spec = importlib.util.spec_from_file_location("kscale_symmetry", _MODULE)
sym = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sym)

HISTORY = 10

JOINT_NAMES = [
    "left_hip_pitch_04", "right_hip_pitch_04",
    "left_hip_roll_03", "right_hip_roll_03",
    "left_hip_yaw_03", "right_hip_yaw_03",
    "left_knee_04", "right_knee_04",
    "left_foot_pitch_02", "right_foot_pitch_02",
    "left_foot_roll_02", "right_foot_roll_02",
]

BODY_NAMES = [
    "assy_formfg___kd_b_102b_torso_btm",
    "kd_d_102l_6061", "kd_d_102r_6061",
    "rs03", "kd_d_201r_6061",
    "kd_d_301l_6061", "kd_d_301r_6061",
    "kd_d_401l_6061", "kd_d_401r_6061",
    "arb_uj111_cross_bearing_2", "arb_uj111_cross_bearing",
    "foot_6061_2", "foot_6061",
]

NJ, NB = len(JOINT_NAMES), len(BODY_NAMES)

POLICY_TERMS = [
    ("base_lin_vel", 3), ("base_ang_vel", 3), ("proj_gravity", 3),
    ("joint_pos", NJ), ("joint_vel", NJ), ("last_action", NJ),
    ("velocity_commands", 3), ("gait_phase", 2),
]

CRITIC_TERMS = POLICY_TERMS[:7] + [
    ("heights", 11 * 17),
    ("robot_joint_torque", NJ), ("robot_joint_acc", NJ),
    ("feet_lin_vel", 6), ("robot_mass", NB), ("robot_inertia", NB * 9),
    ("robot_joint_pos", NJ), ("robot_joint_stiffness", NJ), ("robot_joint_damping", NJ),
    ("robot_pos", 3), ("robot_vel", 6), ("robot_material_properties", NB * 3),
    ("feet_contact_force", 6), ("gait_phase", 2),
]


class _Stub:
    """Minimal stand-ins for the observation manager, the articulation and the environment."""

    class Robot:
        joint_names = JOINT_NAMES
        body_names = BODY_NAMES

    class ObsManager:
        active_terms = {
            "policy": [n for n, _ in POLICY_TERMS],
            "critic": [n for n, _ in CRITIC_TERMS],
        }
        group_obs_term_dim = {
            "policy": [(HISTORY, w) for _, w in POLICY_TERMS],
            "critic": [(HISTORY, w) for _, w in CRITIC_TERMS],
        }

    observation_manager = ObsManager()
    scene = {"robot": Robot()}

    @property
    def unwrapped(self):
        return self


def main() -> int:
    env = _Stub()
    device = torch.device("cpu")
    sym._CACHE.clear()
    maps = sym._maps(env, device)

    failures = 0

    def check(label, ok, detail=""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        if not ok:
            failures += 1

    print("kscale symmetry self-test\n")

    # -- permutations --------------------------------------------------------------------------
    jperm = maps["jperm"].tolist()
    jsign = maps["jsign"].tolist()
    check("joint permutation is an involution", all(jperm[jperm[i]] == i for i in range(NJ)))
    check("joint permutation has no fixed point", all(jperm[i] != i for i in range(NJ)),
          "every joint of a biped has a contralateral partner")
    bperm = maps["bperm"].tolist()
    check("body permutation is an involution", all(bperm[bperm[i]] == i for i in range(NB)))
    check("torso is its own body partner", bperm[0] == 0)

    expected_flip = {"hip_pitch", "hip_roll", "hip_yaw", "foot_roll"}
    got_flip = {
        JOINT_NAMES[i].split("_", 1)[1].rsplit("_", 1)[0]
        for i in range(NJ) if jsign[i] < 0
    }
    check("flip set is the composed set", got_flip == expected_flip, f"{sorted(got_flip)}")

    # -- double mirror is the identity ---------------------------------------------------------
    torch.manual_seed(0)
    n = 8
    for group, terms, kinds in (
        ("policy", POLICY_TERMS, sym._POLICY_KIND),
        ("critic", CRITIC_TERMS, sym._CRITIC_KIND),
    ):
        names = [nm for nm, _ in terms]
        widths = [w * HISTORY for _, w in terms]
        flat = torch.randn(n, sum(widths))
        once = sym._mirror_group(flat, names, widths, kinds, n, maps)
        twice = sym._mirror_group(once, names, widths, kinds, n, maps)
        check(f"{group}: mirroring twice returns the original", torch.allclose(flat, twice, atol=1e-6))
        check(f"{group}: mirroring once changes the tensor", not torch.allclose(flat, once, atol=1e-6),
              "a mirror that changes nothing is the silent-identity failure mode")

    actions = torch.randn(n, NJ)
    twice_a = (actions[:, maps["jperm"]] * maps["jsign"])[:, maps["jperm"]] * maps["jsign"]
    check("actions: mirroring twice returns the original", torch.allclose(actions, twice_a, atol=1e-6))

    # -- the gait pair negates together --------------------------------------------------------
    check("gait phase negates both channels", maps["gait"].tolist() == [-1.0, -1.0],
          "the pair is one shared clock, so the foot swap is a half-cycle shift")

    # -- height grid ---------------------------------------------------------------------------
    check("height grid is lateral rows by forward columns",
          sym._HEIGHT_ROWS * sym._HEIGHT_COLS == 11 * 17,
          "holds only against the CORRECTED root frame")

    print(f"\n{'all checks passed' if failures == 0 else str(failures) + ' CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
