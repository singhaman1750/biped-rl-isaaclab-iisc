"""USD/URDF-based robot design generator for co-optimisation.

This module provides the interface for generating robot design populations
used in evolutionary algorithm-based design and policy co-optimisation.

``param_ranges`` dict keys accepted by :class:`RandomDesignGenerator`:

- ``"thigh_length_scale"``    – scale factor for thigh segment length
                                (Z-offset of knee_L/R_Joint in URDF). Range: (0.85, 1.15)
- ``"shank_length_scale"``    – scale factor for shank segment length
                                (Z-offset of ankle_L/R_Joint in URDF). Range: (0.85, 1.15)
- ``"link_mass_scale"``       – scale factor applied to all link masses and
                                inertia diagonal terms (skips limx_imu). Range: (0.80, 1.20)
- ``"actuator_radius_scale"`` – scale factor for cylinder radius of abad/hip
                                actuator collision geometry. Range: (0.80, 1.20)
- ``"actuator_length_scale"`` – scale factor for cylinder length of abad/hip
                                actuator collision geometry. Range: (0.85, 1.15)
- ``"joint_effort_scale"``    – scale factor for <limit effort> per joint group
                                in the URDF. Range: (0.70, 1.30)
- ``"velocity_limit_scale"``  – scale factor for <limit velocity> per joint group
                                in the URDF. Range: (0.80, 1.20)
- ``"friction_static_scale"`` – scale factor for IdentifiedActuator.friction_static,
                                applied post-respawn. Range: (0.70, 1.40)
- ``"friction_dynamic_scale"``– scale factor for IdentifiedActuator.friction_dynamic,
                                applied post-respawn. Range: (0.70, 1.40)
- ``"saturation_effort_scale"``– scale factor for IdentifiedActuator.saturation_effort,
                                applied post-respawn. Range: (0.70, 1.30)
- ``"armature_scale"``        – scale factor for IdentifiedActuator.armature,
                                applied post-respawn. Range: (0.70, 1.30)

Per-group stiffness/damping scales (applied post-respawn via ``apply_actuator_params``).
Ranges are symmetric equivalents of the runtime domain-randomisation abs ranges;
asymmetric abs ranges are resolved by taking the smaller half (tightest constraint):

- ``"abad_stiffness_scale"``  – IdentifiedActuator.stiffness for abad joints.
                                Derived from abs(50,70) on baseline 55 → ±9.1%.
                                Range: (0.909, 1.091)
- ``"abad_damping_scale"``    – IdentifiedActuator.damping for abad joints.
                                Derived from abs(12,15) on baseline 13.5 → ±11.1%.
                                Range: (0.889, 1.111)
- ``"hip_stiffness_scale"``   – IdentifiedActuator.stiffness for hip joints.
                                Derived from abs(70,90) on baseline 80 → ±12.5%.
                                Range: (0.875, 1.125)
- ``"hip_damping_scale"``     – IdentifiedActuator.damping for hip joints.
                                Derived from abs(10,15) on baseline 13; upper half
                                15.4% is smaller → Range: (0.846, 1.154)
- ``"knee_stiffness_scale"``  – IdentifiedActuator.stiffness for knee joints.
                                Derived from abs(50,70) on baseline 60 → ±16.7%.
                                Range: (0.833, 1.167)
- ``"knee_damping_scale"``    – IdentifiedActuator.damping for knee joints.
                                Derived from abs(3,5) on baseline 4 → ±25%.
                                Range: (0.750, 1.250)
- ``"ankle_stiffness_scale"`` – IdentifiedActuator.stiffness for ankle joints.
                                Derived from abs(8,12) on baseline 10 → ±20%.
                                Range: (0.800, 1.200)
- ``"ankle_damping_scale"``   – IdentifiedActuator.damping for ankle joints.
                                Derived from abs(0.4,0.6) on baseline 0.5 → ±20%.
                                Range: (0.800, 1.200)
"""

from __future__ import annotations

import math
import numpy as np
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

import cma
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from bipedal_locomotion.assets.config.solefoot_identified_cfg import (
    TRON1_ABAD_ACTUATOR_CFG,
    TRON1_ANKLE_ACTUATOR_CFG,
    TRON1_HIP_ACTUATOR_CFG,
    TRON1_KNEE_ACTUATOR_CFG,
)

# ---------------------------------------------------------------------------
# Actuator baseline values (derived from solefoot_identified_cfg.py)
# ---------------------------------------------------------------------------


def _actuator_baseline(cfg) -> dict:
    """Extract baseline scalar values from an IdentifiedActuatorCfg."""
    # TODO check if armature here is collected appropriately
    return {
        "effort_limit": cfg.effort_limit,
        "saturation_effort": cfg.saturation_effort,
        "armature": next(iter(cfg.armature.values())),
        "velocity_limit": cfg.velocity_limit,
        "friction_static": cfg.friction_static,
        "friction_dynamic": cfg.friction_dynamic,
        "stiffness": next(iter(cfg.stiffness.values())),
        "damping": next(iter(cfg.damping.values())),
    }


ACTUATOR_BASELINES: dict[str, dict] = {
    "abad": _actuator_baseline(TRON1_ABAD_ACTUATOR_CFG),
    "hip": _actuator_baseline(TRON1_HIP_ACTUATOR_CFG),
    "knee": _actuator_baseline(TRON1_KNEE_ACTUATOR_CFG),
    "ankle": _actuator_baseline(TRON1_ANKLE_ACTUATOR_CFG),
}

JOINT_TO_ACTUATOR: dict[str, str] = {
    "abad_L_Joint": "abad",
    "abad_R_Joint": "abad",
    "hip_L_Joint": "hip",
    "hip_R_Joint": "hip",
    "knee_L_Joint": "knee",
    "knee_R_Joint": "knee",
    "ankle_L_Joint": "ankle",
    "ankle_R_Joint": "ankle",
}

# Links that belong to abad/hip assemblies (affected by actuator geometry scale)
ABAD_HIP_LINKS: tuple[str, ...] = (
    "abad_L_Link",
    "abad_R_Link",
    "hip_L_Link",
    "hip_R_Link",
)

# Link to skip when scaling mass (IMU is not a structural body)
IMU_LINK_NAME = "limx_imu"

DEFAULT_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "thigh_length_scale": (0.85, 1.15),
    "shank_length_scale": (0.85, 1.15),
    "link_mass_scale": (0.80, 1.20),
    "actuator_radius_scale": (0.80, 1.20),
    "actuator_length_scale": (0.85, 1.15),
    "joint_effort_scale": (0.70, 1.30),
    "velocity_limit_scale": (0.80, 1.20),
    "friction_static_scale": (0.70, 1.40),
    "friction_dynamic_scale": (0.70, 1.40),
    "saturation_effort_scale": (0.70, 1.30),
    "armature_scale": (0.70, 1.30),
    # Per-group stiffness/damping scales derived from symmetric equivalents of the
    # per-group abs ranges in the runtime domain-randomisation configs.
    # Asymmetric abs ranges are resolved by taking the smaller half (tightest constraint).
    "abad_stiffness_scale":  (0.909, 1.091),  # abs(50,70) on baseline 55 → ±9.1%
    "abad_damping_scale":    (0.889, 1.111),  # abs(12,15) on baseline 13.5 → ±11.1%
    "hip_stiffness_scale":   (0.875, 1.125),  # abs(70,90) on baseline 80 → ±12.5%
    "hip_damping_scale":     (0.846, 1.154),  # abs(10,15) on baseline 13 → upper 15.4% (smallest)
    "knee_stiffness_scale":  (0.833, 1.167),  # abs(50,70) on baseline 60 → ±16.7%
    "knee_damping_scale":    (0.750, 1.250),  # abs(3,5) on baseline 4 → ±25%
    "ankle_stiffness_scale": (0.800, 1.200),  # abs(8,12) on baseline 10 → ±20%
    "ankle_damping_scale":   (0.800, 1.200),  # abs(0.4,0.6) on baseline 0.5 → ±20%
}


# ---------------------------------------------------------------------------
# Length-scalable links (used by CMAESDesignGenerator)
# ---------------------------------------------------------------------------

# Constant geometric offset between a scaled link's bottom edge and its
# child joint's local z (see base_robot.urdf line 256 — comment "+ 0.5" is
# a typo for "+ 0.05"; numerically -0.3 = -(0.25 + 0.05) confirms 0.05).
SCALABLE_LINK_CHILD_OFFSET: float = 0.05

# The single child joint whose origin z depends on each scalable link's length.
# This is a structural relationship that doesn't change across URDF edits.
SCALABLE_LINK_CHILD_JOINTS: dict[str, str] = {
    "hip_R_thigh_Link": "knee_R_Joint",
    "hip_L_thigh_Link": "knee_L_Joint",
    "knee_R_Link":      "ankle_R_actuator_Joint",
    "knee_L_Link":      "ankle_L_actuator_Joint",
}


def _parse_scalable_links_from_urdf(
    urdf_path: str,
) -> tuple[dict[str, dict], dict[str, float]]:
    """Parse box size and mass for each scalable link from the base URDF.

    Returns:
        scalable_links: dict mapping link name to {"size": (x,y,z), "mass": m,
                        "child_joint": joint_name}.
        link_densities: dict mapping link name to density (kg/m³).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    scalable_links: dict[str, dict] = {}
    for link_name, child_joint in SCALABLE_LINK_CHILD_JOINTS.items():
        link_el = None
        for link in root.iter("link"):
            if link.get("name") == link_name:
                link_el = link
                break
        if link_el is None:
            raise ValueError(
                f"Scalable link {link_name!r} not found in {urdf_path}"
            )

        # Read box size from <visual>/<geometry>/<box>
        box = link_el.find("visual/geometry/box")
        if box is None:
            raise ValueError(
                f"Link {link_name!r} has no <visual>/<geometry>/<box> in {urdf_path}"
            )
        sx, sy, sz = (float(v) for v in box.get("size").split())

        # Read mass from <inertial>/<mass>
        mass_el = link_el.find("inertial/mass")
        if mass_el is None:
            raise ValueError(
                f"Link {link_name!r} has no <inertial>/<mass> in {urdf_path}"
            )
        mass = float(mass_el.get("value"))

        scalable_links[link_name] = {
            "size": (sx, sy, sz),
            "mass": mass,
            "child_joint": child_joint,
        }

    link_densities = {
        name: meta["mass"] / (meta["size"][0] * meta["size"][1] * meta["size"][2])
        for name, meta in scalable_links.items()
    }
    return scalable_links, link_densities

# CMA-ES restricts its search to the two length scales it actually drives.
CMAES_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "thigh_length_scale": DEFAULT_PARAM_RANGES["thigh_length_scale"],
    "shank_length_scale": DEFAULT_PARAM_RANGES["shank_length_scale"],
}


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class Population(ABC):
    """Abstract base class for a design population.

    A population holds a fixed set of robot designs (individuals).  Each
    individual is represented by a USD file path and an actuator-parameter
    dict that overrides the Python-side ``IdentifiedActuator`` attributes.
    """

    @abstractmethod
    def get_usd_files(self) -> list[str]:
        """Return a list of USD file paths, one per individual."""
        ...

    @abstractmethod
    def get_actuator_params(self) -> list[dict[str, dict]]:
        """Return actuator override dicts, one per individual.

        Each element is a dict keyed by actuator group name
        (``"abad"``, ``"hip"``, ``"knee"``, ``"ankle"``), whose value is a
        dict of scalar overrides for the ``IdentifiedActuator`` tensor
        attributes (e.g. ``{"effort_limit": 45.0, "friction_static": 0.25}``).
        """
        ...


class DesignGeneratorBase(ABC):
    """Abstract base class for design generators.

    A design generator encapsulates an optimisation algorithm (e.g. random
    search, evolutionary algorithm) and produces successive :class:`Population`
    objects.  The runner calls :meth:`generate_population` at the start of
    every EA generation, and :meth:`update_with_fitness` after evaluating the
    current population so that the generator can improve future generations.
    """

    @abstractmethod
    def generate_population(self, generation: int) -> Population:
        """Generate a new population for *generation*."""
        ...

    def update_with_fitness(self, fitness: list[float]) -> None:
        """Optionally update internal state using per-individual fitness scores.

        The default implementation is a no-op (random search).  Override this
        to implement selection, mutation, cross-over, etc.

        Args:
            population: The population that was just evaluated.
            fitness: Mean episode return for each individual, indexed by
                individual index (not env index).
        """
        pass

    def sample_batch() -> None:
        pass


class RandomPopulation(Population):
    """A concrete population backed by pre-generated USD paths and actuator params."""

    def __init__(
        self, usd_files: list[str], actuator_params: list[dict[str, dict]]
    ) -> None:
        self._usd_files = usd_files
        self._actuator_params = actuator_params

    def get_usd_files(self) -> list[str]:
        return self._usd_files

    def get_actuator_params(self) -> list[dict[str, dict]]:
        return self._actuator_params


class RandomDesignGenerator(DesignGeneratorBase):
    """Generates robot design populations by randomly perturbing a base URDF.

    Each individual is produced by:

    1. Sampling scalar scale factors from ``param_ranges``.
    2. Modifying the parsed URDF tree (joint origins, link masses/inertias,
       actuator cylinder geometry, joint limits).
    3. Writing the modified URDF to ``output_dir``.
    4. Converting the URDF to USD via :class:`isaaclab.sim.converters.UrdfConverter`.
    5. Recording actuator-param overrides that cannot be expressed in USD/URDF.

    Args:
        base_urdf_path: Absolute path to the template URDF file.
        num_individuals: Number of designs per generation.
        param_ranges: Optional dict overriding entries in
            :data:`DEFAULT_PARAM_RANGES`.  Only the keys you provide are
            overridden; all other defaults remain.
        output_dir: Directory for writing generated URDFs and USDs.
    """

    def __init__(
        self,
        base_urdf_path: str,
        num_individuals: int,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        output_dir: str = "/tmp/copt_usds",
    ) -> None:
        self.base_urdf_path = base_urdf_path
        self.num_individuals = num_individuals
        self.output_dir = output_dir

        # Merge user overrides with defaults
        self.param_ranges: dict[str, tuple[float, float]] = {**DEFAULT_PARAM_RANGES}
        if param_ranges is not None:
            self.param_ranges.update(param_ranges)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_population(self, generation: int) -> Population:
        """Generate *num_individuals* robot designs for *generation*.

        Returns a :class:`RandomPopulation` with USD paths and actuator params.
        """
        usd_files: list[str] = []
        actuator_params: list[dict[str, dict]] = []

        for idx in range(self.num_individuals):
            usd_path, act_params = self._generate_individual(generation, idx)
            usd_files.append(usd_path)
            actuator_params.append(act_params)

        return RandomPopulation(usd_files, actuator_params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_scales(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            key: float(rng.uniform(lo, hi))
            for key, (lo, hi) in self.param_ranges.items()
        }

    def _generate_individual(
        self, generation: int, idx: int
    ) -> tuple[str, dict[str, dict]]:
        """Build one perturbed URDF + USD and return (usd_path, actuator_params)."""
        # Deterministic per (generation, individual) pair
        rng = np.random.default_rng(seed=generation * 10000 + idx)
        scales = self._sample_scales(rng)
        print('generating individual with the following scales')
        for key, item in scales.enumerate():
            print(f'{key}: {item}')

        # ---- Parse template URDF ----------------------------------------
        tree = ET.parse(self.base_urdf_path)
        root = tree.getroot()

        # ---- B: Thigh / shank length ------------------------------------
        s_thigh = scales["thigh_length_scale"]
        s_shank = scales["shank_length_scale"]
        self._scale_joint_z_origin(root, ["knee_L_Joint", "knee_R_Joint"], s_thigh)
        self._scale_joint_z_origin(root, ["ankle_L_Joint", "ankle_R_Joint"], s_shank)

        # ---- C: Link mass & inertia -------------------------------------
        s_mass = scales["link_mass_scale"]
        for link in root.iter("link"):
            link_name = link.get("name", "")
            if IMU_LINK_NAME in link_name:
                continue
            for mass_el in link.iter("mass"):
                v = float(mass_el.get("value", 0.0))
                mass_el.set("value", str(v * s_mass))
            for inertia_el in link.iter("inertia"):
                for attr in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
                    val = inertia_el.get(attr)
                    if val is not None:
                        inertia_el.set(attr, str(float(val) * s_mass))

        # ---- D: Actuator cylinder geometry (abad/hip links) -------------
        s_r = scales["actuator_radius_scale"]
        s_l = scales["actuator_length_scale"]
        for link in root.iter("link"):
            link_name = link.get("name", "")
            if any(al in link_name for al in ABAD_HIP_LINKS):
                for cylinder in link.iter("cylinder"):
                    r = float(cylinder.get("radius", 0.0))
                    l_val = float(cylinder.get("length", 0.0))
                    cylinder.set("radius", str(r * s_r))
                    cylinder.set("length", str(l_val * s_l))

        # ---- E: Joint effort & velocity limits --------------------------
        s_eff = scales["joint_effort_scale"]
        s_vel = scales["velocity_limit_scale"]
        for joint in root.iter("joint"):
            joint_name = joint.get("name", "")
            actuator_group = JOINT_TO_ACTUATOR.get(joint_name)
            if actuator_group is None:
                continue
            baseline = ACTUATOR_BASELINES[actuator_group]
            for limit_el in joint.iter("limit"):
                limit_el.set("effort", str(baseline["effort_limit"] * s_eff))
                limit_el.set("velocity", str(baseline["velocity_limit"] * s_vel))

        # ---- F: Write modified URDF -------------------------------------
        gen_dir = os.path.join(self.output_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        urdf_path = os.path.join(gen_dir, f"individual_{idx:04d}.urdf")
        tree.write(urdf_path, xml_declaration=True, encoding="utf-8")

        # ---- G: Convert URDF → USD --------------------------------------
        usd_out_dir = os.path.join(gen_dir, f"individual_{idx:04d}_usd")
        usd_path = self._convert_urdf_to_usd(urdf_path, usd_out_dir, idx)

        # ---- H: Build actuator params dict (Python-side attrs) ----------
        s_fs = scales["friction_static_scale"]
        s_fd = scales["friction_dynamic_scale"]
        s_sat = scales["saturation_effort_scale"]
        s_arm = scales["armature_scale"]

        act_params: dict[str, dict] = {}
        for group, baseline in ACTUATOR_BASELINES.items():
            s_stiff = scales[f"{group}_stiffness_scale"]
            s_damp = scales[f"{group}_damping_scale"]
            act_params[group] = {
                "effort_limit": baseline["effort_limit"] * s_eff,
                "velocity_limit": baseline["velocity_limit"] * s_vel,
                "saturation_effort": baseline["saturation_effort"] * s_sat,
                "armature": baseline["armature"] * s_arm,
                "friction_static": baseline["friction_static"] * s_fs,
                "friction_dynamic": baseline["friction_dynamic"] * s_fd,
                "stiffness": baseline["stiffness"] * s_stiff,
                "damping": baseline["damping"] * s_damp,
            }

        return usd_path, act_params

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_joint_z_origin(
        root: ET.Element, joint_names: list[str], scale: float
    ) -> None:
        """Scale the Z component of the <origin xyz="..."> of specified joints."""
        for joint in root.iter("joint"):
            if joint.get("name") not in joint_names:
                continue
            for origin in joint.iter("origin"):
                xyz_str = origin.get("xyz", "0 0 0")
                x, y, z = (float(v) for v in xyz_str.split())
                origin.set("xyz", f"{x} {y} {z * scale}")

    @staticmethod
    def _convert_urdf_to_usd(urdf_path: str, usd_out_dir: str, idx: int) -> str:
        """Converts a URDF file to USD using IsaacLab's UrdfConverter."""
        usd_file_name = f"biped_{idx}.usd"
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=usd_out_dir,
            usd_file_name=usd_file_name,
            link_density=0.0,
            merge_fixed_joints=False,
            fix_base=False,
            self_collision=False,
            collider_type="convex_hull",
            joint_drive=None,
            force_usd_conversion=True,
        )
        converter = UrdfConverter(cfg)
        return converter.usd_path

class CMAESDesignGenerator(RandomDesignGenerator):
    """Generates robot designs by sampling from a CMA-ES search distribution.

    The design vector is the unit-hypercube encoding of the scale-factor
    dictionary used by :class:`RandomDesignGenerator`; CMA-ES adapts the
    multivariate Gaussian search distribution from the per-individual
    mean episode return reported by :class:`CoptOnPolicyRunner`.
    """

    def __init__(
        self,
        base_urdf_path: str,
        num_individuals: int,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        output_dir: str = "/tmp/copt_usds",
        sigma0: float = 0.2,
        seed: int = 0,
        es_state_path: str | None = None,
        late_start: bool = False
    ) -> None:
        super().__init__(
            base_urdf_path=base_urdf_path,
            num_individuals=num_individuals,
            param_ranges=param_ranges,
            output_dir=output_dir,
        )
        # Restrict CMA-ES to only the length scales this generator drives.
        # Caller-supplied param_ranges may tune the bounds of these two keys
        # but cannot add new dimensions (any extras would be inert).
        self.param_ranges = {**CMAES_PARAM_RANGES}
        if param_ranges is not None:
            for key, rng in param_ranges.items():
                if key in self.param_ranges:
                    self.param_ranges[key] = rng

        self.param_keys: list[str] = list(self.param_ranges.keys())
        self.num_params: int = len(self.param_keys)
        self._sigma0: float = float(sigma0)
        self._seed: int = int(seed)
        self.late_start = late_start

        if es_state_path is not None:
            with open(es_state_path, "rb") as fh:
                self._es = cma.CMAEvolutionStrategy.pickle_loads(fh.read())
            assert self._es.N == self.num_params, (
                f"Checkpoint dimension mismatch: "
                f"pickled {self._es.N}, expected {self.num_params}"
            )
        else:
            opts = cma.CMAOptions()
            opts.set({
                "popsize": int(num_individuals),
                "bounds": [np.zeros(self.num_params), np.ones(self.num_params)],
                "seed": self._seed,
                "verb_disp": 1,
                "verb_filenameprefix": str(Path(output_dir) / "cma_log") + "/",
            })
            self._es = cma.CMAEvolutionStrategy(
                np.full(self.num_params, 1.0), self._sigma0, opts
            )

        self._pending_solutions: list[np.ndarray] | None = None
        self._last_solutions: list[np.ndarray] | None = None
        self._terminated = False

        # Parse box sizes and masses from the base URDF so they always mirror
        # the actual URDF values rather than relying on hardcoded constants.
        self.scalable_links, self.link_densities = (
            _parse_scalable_links_from_urdf(base_urdf_path)
        )

        # Holds the parsed URDF root for the duration of one _generate_individual call.
        self._current_root: ET.Element | None = None

    # --- Public API ---------------------------------------------------

    def toggle_late_start(self) -> None:
        self.late_start = not self.late_start

    def sample_batch(self) -> None:
        assert self._pending_solutions is None, "sample_batch called twice consecutively"
        self._pending_solutions = self._es.ask()

    def update_with_fitness(
        self, fitness: list[float]
    ) -> None:
        assert self._last_solutions is not None, (
            "update_with_fitness called before generate_population"
        )
        if not self._terminated:
            print("Updating Designs with PPO Training Roll")
            costs = [self._sanitise_cost(-float(f)) for f in fitness]
            self._es.tell(self._last_solutions, costs)
            if self._es.stop():
                print(f"[CMA-ES] terminated: {self._es.stop()}")
                self._terminated = True
            self.sample_batch()

    def generate_population(self, generation: int) -> Population:
        if not self._terminated:
            print("Generating New Designs using CMAES output")
            assert self._pending_solutions is not None, (
                "generate_population called before sample_batch"
            )
            usd_files: list[str] = []
            actuator_params: list[dict[str, dict]] = []
            print('generating individual with the following scales')
            for idx, _ in enumerate(self._pending_solutions):
                usd_path, act_params = self._generate_individual(generation, idx)
                usd_files.append(usd_path)
                actuator_params.append(act_params)
            self._last_solutions = self._pending_solutions
            self._pending_solutions = None
            return RandomPopulation(usd_files, actuator_params)

    # --- Internal helpers --------------------------------------------

    def _denormalise(self, x: np.ndarray) -> dict[str, float]:
        scales: dict[str, float] = {}
        for i, key in enumerate(self.param_keys):
            lo, hi = self.param_ranges[key]
            xi = float(np.clip(x[i], 0.0, 1.0))
            scales[key] = float(lo + xi * (hi - lo))
        return scales

    @staticmethod
    def _sanitise_cost(c: float, fallback: float = 1e6) -> float:
        if not np.isfinite(c):
            return fallback
        return c

    def save_state(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self._es.pickle_dumps())

    def _sample_scales(self, rng: np.random.Generator) -> dict[str, float]:
        return super()._sample_scales(rng)

    def _generate_individual(
        self, generation: int, idx: int
    ) -> tuple[str, dict[str, dict]]:
        """Build one length-scaled URDF + USD and return (usd_path, {})."""
        scales = None
        if self.late_start:
            rng = np.random.default_rng(seed=generation * 10000 + idx)
            scales = self._sample_scales(rng)
        else:
            scales = self._denormalise(self._pending_solutions[idx])
        print("===============================================")
        print(f'late start: {self.late_start}')
        for key, item in scales.items():
            print(f'{key}: {item}')

        s_thigh = scales["thigh_length_scale"]
        s_shank = scales["shank_length_scale"]

        tree = ET.parse(self.base_urdf_path)
        self._current_root = tree.getroot()

        self._update_link_length("hip_R_thigh_Link", s_thigh)
        self._update_link_length("hip_L_thigh_Link", s_thigh)
        self._update_link_length("knee_R_Link", s_shank)
        self._update_link_length("knee_L_Link", s_shank)

        gen_dir = os.path.join(self.output_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        urdf_path = os.path.join(gen_dir, f"individual_{idx:04d}.urdf")
        tree.write(urdf_path, xml_declaration=True, encoding="utf-8")

        usd_out_dir = os.path.join(gen_dir, f"individual_{idx:04d}_usd")
        usd_path = self._convert_urdf_to_usd(urdf_path, usd_out_dir, idx)

        self._current_root = None
        # Actuator overrides intentionally empty for now; structure preserved
        # so downstream _reload_morphology / RandomPopulation consumers don't break.
        return usd_path, {}

    # ------------------------------------------------------------------
    # Length-scaling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_box_mass(density: float, x: float, y: float, z: float) -> float:
        return density * x * y * z

    @staticmethod
    def _get_box_inertia(
        mass: float, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        """Diagonal MoI of a solid box at its CoM (per base_robot.md)."""
        ixx = mass * (y * y + z * z) / 12.0
        iyy = mass * (x * x + z * z) / 12.0
        izz = mass * (x * x + y * y) / 12.0
        return ixx, iyy, izz

    def _update_inertial(
        self, link_name: str, mass: float, moi: tuple[float, float, float]
    ) -> None:
        """Set <mass> and <inertia> on the named link. Off-diagonals are zeroed."""
        link = self._find_link(link_name)
        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(f"Link {link_name!r} has no <inertial> element.")

        mass_el = inertial.find("mass")
        if mass_el is None:
            raise ValueError(f"Link {link_name!r}/<inertial> has no <mass>.")
        mass_el.set("value", str(mass))

        inertia_el = inertial.find("inertia")
        if inertia_el is None:
            raise ValueError(f"Link {link_name!r}/<inertial> has no <inertia>.")
        ixx, iyy, izz = moi
        inertia_el.set("ixx", str(ixx))
        inertia_el.set("iyy", str(iyy))
        inertia_el.set("izz", str(izz))
        inertia_el.set("ixy", "0")
        inertia_el.set("ixz", "0")
        inertia_el.set("iyz", "0")

    def _update_joint_position(
        self, link_name: str, joint_name: str, z: float
    ) -> None:
        """Set the z component of <joint name=joint_name>/<origin xyz>.

        Asserts the joint's <parent link> matches `link_name` so only joints
        whose parent is the scaled link are touched.
        """
        joint = self._find_joint(joint_name)
        parent = joint.find("parent")
        parent_link = parent.get("link") if parent is not None else None
        if parent_link != link_name:
            raise ValueError(
                f"Joint {joint_name!r} parent is {parent_link!r}, expected "
                f"{link_name!r}; refusing to update."
            )
        origin = joint.find("origin")
        if origin is None:
            raise ValueError(f"Joint {joint_name!r} has no <origin> element.")
        x, y, _ = (float(v) for v in origin.get("xyz", "0 0 0").split())
        origin.set("xyz", f"{x} {y} {z}")

    def _update_link_length(self, link_name: str, scale: float) -> None:
        """Scale the z dimension of a self.scalable_links entry and propagate.

        Updates the link's box geometry, element origins, mass+inertia, and
        the single child joint whose parent is the scaled link.
        """
        if link_name not in self.scalable_links:
            raise ValueError(f"{link_name!r} is not a scalable link.")
        meta = self.scalable_links[link_name]
        x, y, z0 = meta["size"]
        child_joint = meta["child_joint"]
        density = self.link_densities[link_name]

        z_new = z0 * scale
        new_origin_z = -z_new / 2.0

        link = self._find_link(link_name)

        # 1) box sizes on <visual> and <collision>
        for el_tag in ("visual", "collision"):
            el = link.find(el_tag)
            if el is None:
                continue
            box = el.find("geometry/box")
            if box is None:
                raise ValueError(
                    f"{link_name}/{el_tag} has no <box> geometry."
                )
            box.set("size", f"{x} {y} {z_new}")
        # print(f"updating {link_name} dimensions from {x},{y},{z0} to {x},{y},{z_new}")
        # print(f"new link length is {math.sqrt(x*x + y * y + z_new * z_new)}")

        # 2) origin z on <visual>, <collision>, <inertial>
        for el_tag in ("visual", "collision", "inertial"):
            el = link.find(el_tag)
            if el is None:
                continue
            origin = el.find("origin")
            if origin is None:
                continue
            ox, oy, _ = (float(v) for v in origin.get("xyz", "0 0 0").split())
            origin.set("xyz", f"{ox} {oy} {new_origin_z}")

        # 3) mass + inertia
        m_new = self._get_box_mass(density, x, y, z_new)
        moi_new = self._get_box_inertia(m_new, x, y, z_new)
        self._update_inertial(link_name, m_new, moi_new)

        # 4) child joint position
        joint_z = -(z_new + SCALABLE_LINK_CHILD_OFFSET)
        self._update_joint_position(link_name, child_joint, joint_z)

    # ------------------------------------------------------------------
    # Element lookup helpers
    # ------------------------------------------------------------------

    def _find_link(self, name: str) -> ET.Element:
        if self._current_root is None:
            raise RuntimeError("_find_link called with no current URDF root.")
        for link in self._current_root.iter("link"):
            if link.get("name") == name:
                return link
        raise KeyError(f"No <link name={name!r}> in current URDF.")

    def _find_joint(self, name: str) -> ET.Element:
        if self._current_root is None:
            raise RuntimeError("_find_joint called with no current URDF root.")
        for joint in self._current_root.iter("joint"):
            if joint.get("name") == name:
                return joint
        raise KeyError(f"No <joint name={name!r}> in current URDF.")
