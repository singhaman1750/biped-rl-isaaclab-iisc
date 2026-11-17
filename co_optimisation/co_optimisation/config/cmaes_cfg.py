"""Configuration of the design search and the co-optimisation loop of a run.

Dumped to <experiment_dir>/params/cmaes.yaml and cmaes.pkl by scripts/rsl_rl/train.py, and
restored from cmaes.pkl when a run is resumed, so that a run continues under the configuration
it began with. See plans/copt_training_refinement.md.
"""

from __future__ import annotations

import copy

from isaaclab.utils import configclass


@configclass
class CMAESCfg:
    """Every setting of the design search and of the runner's co-optimisation loop.

    The values are those every djinn launch of a co-optimisation run used before 2026-10-28.
    The tolerances and the fine-tune are None, so CMA-ES keeps pycma's defaults and the run
    trains to max_iterations. The robot subclasses set both.
    """

    design_search: str = "leg_geometry"
    """The co-optimisation type, one of three. "leg_geometry" searches the link length scales and the
    nominal pose offsets, by CMA-ES after a random phase whose prior widens from the donor to the full
    box. "actuator_leg_geometry" searches the same together with the actuator of every joint group, by
    CatCMAwM. "actuator" searches the actuator of every joint group together with the gain scales
    param_ranges names, by CatCMAwM on the baseline's link lengths and pose, and requires param_ranges
    to name gain scales alone, at least one, which train.py checks."""

    num_individuals: int = 80
    """Designs per generation, which must divide the scene's num_envs."""

    ea_update_interval: int = 400
    """Policy iterations between design generations."""

    ea_late_start: int = 6000
    """Iteration at which the random phase hands over to CMA-ES."""

    param_ranges: dict = {
        "thigh_length_scale": ("scale", (0.75, 1.25)),
        "shank_length_scale": ("scale", (0.75, 1.25)),
        "thigh_bend_offset": ("offset", (-0.10, 0.10)),
        "knee_bend_offset": ("offset", (-0.25, 0.25)),
    }
    """The searched parameters and their ranges, in the form usd_generator.DEFAULT_PARAM_RANGES takes.
    Its keys alone decide what is searched, a geometry key left out keeping the baseline's value. A gain
    scale of usd_generator.GAIN_PARAM_KEYS, such as "knee_damping_scale", is searched and applied to the
    actuators where named, and one naming no joint group of the robot raises."""

    sigma0: float = 0.25
    """Initial step size of CMA-ES in the unit box."""

    seed: int = 42
    """Seed of the CMA-ES strategy."""

    late_start_prior_divisor: int = 64
    """The random phase draws num_envs // late_start_prior_divisor designs per generation."""

    max_cma_iter: int | None = None
    """Generation cap of CMA-ES. None derives (max_iterations - ea_late_start) / ea_update_interval."""

    tolx: float | None = None
    """Largest sampling std in the unit box at which CMA-ES stops. None keeps pycma's 1e-11."""

    tolfun: float | None = None
    """Current and historic fitness range at which CMA-ES stops. None keeps pycma's 1e-11."""

    tolfunhist: float | None = None
    """Range of the best fitness over the last 10 + 30 N / popsize generations at which CMA-ES stops.
    None keeps pycma's 1e-12."""

    finetune_iterations: int | None = None
    """Iterations trained on the distribution mean once CMA-ES terminates, after which training
    ends. None trains on to max_iterations on the last population."""

    tolcat: float | None = None
    """The categorical searches only, the largest gap between each actuator group's likeliest pick and
    the probability CatCMAwM's margin lets it reach at which the actuator choice counts as settled, which
    with tolx stops the search. None never counts it settled, leaving maxiter, tolfun and tolfunhist."""

    randomise_before_late_start: bool = True
    """Swap the design population at every generation of the random phase."""

    reinit_lr_each_generation: bool = True
    """Re-initialise the learning rate at every generation of the random phase."""

    eval_duration_s: float = 4.0
    """Horizon of the controlled evaluation of a generation."""

    eval_seed: int = 42
    """Seed of the Latin hypercube of commands the evaluation draws."""

    shank_foot_gap: float | None = None
    """Quadruped only, the gap in metres between each shank's collision box and its foot sphere.
    None keeps the box reaching the sphere's centre."""

    filters: list = []
    """The design filters every population passes before it is converted, each a key of
    co_optimisation.utils.filters.FILTERS, such as "StandingFilter", or a "module:Class" path to a
    FilterBase subclass. Empty, the default, filters nothing."""

    filter_params: dict = {}
    """Keyword arguments of each filter, keyed as filters names it. A filter left out takes its defaults."""


@configclass
class BipedCMAESCfg(CMAESCfg):
    """The TRON1 solefoot biped, tasks Isaac-Limx-SF-Copt-*. Tolerances of plans/copt_training_refinement.md section 3.9,
    and the standing filter at the defaults of plans/design_filtering.md section 2.5, written out so that
    params/cmaes.yaml records them."""

    tolx: float | None = 0.02
    tolfun: float | None = 0.8
    tolfunhist: float | None = 0.8
    finetune_iterations: int | None = 400
    tolcat: float | None = 0.05
    filters: list = ["StandingFilter"]
    filter_params: dict = {
        "StandingFilter": {
            "num_positions": 16,
            "num_torques": 16,
            "max_failure_fraction": 0.25,
            "position_fraction": 0.25,
            "torque_fraction": [0.5, 1.0],
            "soft_limit_factor": 0.9,
            "seed": 0,
        },
    }

    design_search: str = "actuator_leg_geometry"

    num_individuals: int = 64


@configclass
class QuadrupedCMAESCfg(CMAESCfg):
    """The quadruped, tasks Isaac-Quadruped-Copt-*. Tolerances of plans/copt_training_refinement.md section 3.9."""

    tolx: float | None = 0.035
    tolfun: float | None = 0.15
    tolfunhist: float | None = 0.15
    finetune_iterations: int | None = 400
    tolcat: float | None = 0.05
    shank_foot_gap: float | None = 0.001
    param_ranges: dict = {
        "thigh_length_scale": ("scale", (0.5, 1.50)),
        "shank_length_scale": ("scale", (0.5, 1.50)),
        "thigh_bend_offset": ("offset", (-0.50, 0.50)),
        "knee_bend_offset": ("offset", (-0.5, 0.5)),
    }

    ea_late_start: int = 7500

    design_search: str = "actuator_leg_geometry"


# The configuration of each design generator family, as scripts/rsl_rl/train.py resolves the family from the task.
CMAES_CFG_BY_FAMILY: dict[str, type[CMAESCfg]] = {
    "biped": BipedCMAESCfg,
    "quadruped": QuadrupedCMAESCfg,
}


def without_termination(cfg: CMAESCfg) -> CMAESCfg:
    """A copy of *cfg* without the tolerances and the fine-tune, for a run begun before params/cmaes.pkl existed."""
    legacy = copy.deepcopy(cfg)
    legacy.tolx = None
    legacy.tolfun = None
    legacy.tolfunhist = None
    legacy.finetune_iterations = None
    return legacy
