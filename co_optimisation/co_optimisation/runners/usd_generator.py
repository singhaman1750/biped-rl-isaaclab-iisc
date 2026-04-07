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
"""

from __future__ import annotations

import numpy as np
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

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

    def update_with_fitness(self, population: Population, fitness: list[float]) -> None:
        """Optionally update internal state using per-individual fitness scores.

        The default implementation is a no-op (random search).  Override this
        to implement selection, mutation, cross-over, etc.

        Args:
            population: The population that was just evaluated.
            fitness: Mean episode return for each individual, indexed by
                individual index (not env index).
        """
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
            act_params[group] = {
                "effort_limit": baseline["effort_limit"] * s_eff,
                "velocity_limit": baseline["velocity_limit"] * s_vel,
                "saturation_effort": baseline["saturation_effort"] * s_sat,
                "armature": baseline["armature"] * s_arm,
                "friction_static": baseline["friction_static"] * s_fs,
                "friction_dynamic": baseline["friction_dynamic"] * s_fd,
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
