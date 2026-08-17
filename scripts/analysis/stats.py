"""Statistics for legged locomotion play dumps.

Computes the descriptive and derived statistics by which a legged locomotion policy is
evaluated and compared against another, from the numpy dump that scripts/rsl_rl/play.py
writes. Pure numpy, with no dependency on Isaac Lab, on torch or on the task package, so
that it runs inside the simulation container and equally in a plain interpreter against a
stored dump, which is what makes it testable without a simulator.

Imported by scripts/rsl_rl/play.py and by nothing else. The record set is computed once,
at the close of a play, by the process that holds the arrays it describes, and written to
statistics.npy beside dump.npy and rewards.npy. Every downstream consumer reads that file
and none recomputes from it, so a figure cannot drift from the data it was taken from.

The output is a flat list of records, each carrying a group, a quantity, a statistic, a
value and a unit, which pivots directly into a table indexed by quantity and statistic.

Provenance. Every reference implementation here was validated against the play dumps of
sd_brs1_flat/2026-07-28_06-37-24 and sd_brs1_flat/2026-08-03_11-19-11 and reproduces the
figures recorded in the twentieth, twenty first and twenty second passes of
/ws/context/brs_gait.md. The validated correspondences are tabulated in
/ws/context/gait_metrics.md and are the regression gate for any change here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Keys of the dump that carry index metadata or a constant of the run rather than a per
# step series. They are written once and stored flat, so a consumer reads
# dump["joint_names"] as the list of names itself. Every other key is a stacked array of
# shape (T, E, ...). Declared here rather than in play.py because the writer and this
# module must agree on it exactly, and two copies of a format definition diverge silently.
METADATA_KEYS = (
    "joint_names",
    "body_names",
    "feet_names",
    "joint_effort_limits",
    "joint_velocity_limits",
    # The solver's ceilings, kept beside the actuator's since 2026-08-07. The two keys
    # above now carry the ACTUATOR ceilings, which are the values a torque is clipped
    # against and therefore the only ones a saturation statistic may be computed from.
    # Until that date they carried these, which for an explicit actuator model are an
    # unbounded default and made the whole saturation family inert.
    "joint_effort_limits_sim",
    "joint_velocity_limits_sim",
    "joint_position_limits",
    "joint_soft_position_limits",
    "default_joint_pos",
    "body_masses",
    "step_dt",
    "contact_force_history_length",
    # Which frame each per body vector channel is expressed in. Absent on any dump
    # written before 2026-08-07, under which the three feet channels are world frame.
    "frame_convention",
)

# Percentiles reported for every scalar summary. The request names these six and the
# record uses the 99th throughout, the traces being impulse trains whose mean understates
# the demand by an order of magnitude, so the upper tail is where the information is.
PERCENTILES = (5, 25, 50, 75, 90, 99)

# Force in newtons above which a body counts as in contact. The record establishes that
# this constant is not incidental. The twentieth pass used a different value and reported
# a touchdown velocity of 1.68 m/s where the twenty first, at 1.0 N, reports 1.551 for the
# same run. Any figure quoted from this module must therefore quote the threshold with it.
CONTACT_FORCE_THRESHOLD = 1.0

# Window, in seconds, over which the peak contact force following a touchdown is taken.
# 0.08 s is the value the twenty first and twenty second passes used and is long enough to
# span the impact transient while excluding the steady stance load that follows it.
IMPACT_WINDOW_S = 0.08

# Tolerance, in radians, within which a joint counts as standing at a mechanical limit,
# and the fraction of the effort ceiling above which an actuator counts as saturated.
# Both are the values of the twentieth pass per joint table.
LIMIT_TOLERANCE_RAD = 0.02
EFFORT_SATURATION_FRACTION = 0.98

GRAVITY = 9.81


@dataclass
class GaitData:
    """A play dump, resolved into named arrays.

    Every per step channel has shape (T, E, ...) where T is the number of control steps
    and E the number of environments, which is the shape the dump itself carries, so
    nothing here reshapes or reindexes what it reads. Optional channels are None where the
    robot or the task does not supply them, a sole clearance on a point footed robot or a
    gait command on a task with no clock, and every consumer must test for that rather
    than assume presence.
    """

    dt: float
    joint_names: list[str]
    body_names: list[str]
    feet_names: list[str]

    joint_positions: np.ndarray          # (T, E, J)
    joint_velocities: np.ndarray         # (T, E, J)
    joint_torques: np.ndarray            # (T, E, J)
    joint_accelerations: np.ndarray      # (T, E, J)
    joint_powers: np.ndarray             # (T, E, J)

    base_com_position: np.ndarray        # (T, E, 3), world frame
    base_linear_velocity: np.ndarray     # (T, E, 3), base frame
    base_angular_velocity: np.ndarray    # (T, E, 3), base frame
    commanded_linear_velocity: np.ndarray   # (T, E, 2), base frame
    commanded_angular_velocity: np.ndarray  # (T, E), yaw rate

    feet_contact_forces: np.ndarray      # (T, E, F, 3), world frame
    feet_velocities: np.ndarray          # (T, E, F, 3), world frame
    feet_frame_heights: np.ndarray       # (T, E, F), world frame

    body_mass: float                     # total, kilograms

    # Optional, absent where the robot, the task or the environment supplies no such
    # quantity, a sole on a point foot, a clock on a task without one, a termination
    # manager on an environment that carries none.
    base_quaternion: np.ndarray | None = None            # (T, E, 4), wxyz
    base_projected_gravity: np.ndarray | None = None     # (T, E, 3), base frame
    feet_quaternions: np.ndarray | None = None           # (T, E, F, 4), wxyz
    feet_sole_clearances: np.ndarray | None = None       # (T, E, F)
    feet_distance: np.ndarray | None = None              # (T, E, 3), signed, foot0 - foot1
    gait_command: np.ndarray | None = None               # (T, E, 4), freq/offset/dur/height
    episode_dones: np.ndarray | None = None              # (T, E), bool
    episode_terminated: np.ndarray | None = None         # (T, E), bool
    episode_time_outs: np.ndarray | None = None          # (T, E), bool

    # Which frame each per body vector channel is expressed in, as written by play.py
    # from 2026-08-07. None on any earlier dump, under which the three feet channels are
    # world frame and their horizontal components must not be read per axis.
    frame_convention: dict | None = None
    joint_effort_limits: np.ndarray | None = None        # (J,)
    joint_velocity_limits: np.ndarray | None = None      # (J,)
    joint_position_limits: np.ndarray | None = None      # (J, 2)
    joint_soft_position_limits: np.ndarray | None = None # (J, 2)
    default_joint_pos: np.ndarray | None = None          # (J,)

    # Depth of the contact sensor's force history, in physics steps, from which the
    # provenance of feet_contact_forces follows. A value above zero means the channel
    # carries the force at the history peak, which is the quantity the impact reward
    # prices.
    contact_force_history_length: int = 0

    rewards: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (T, E) RATE
    reward_weights: dict[str, float] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return self.joint_positions.shape[0]

    @property
    def num_envs(self) -> int:
        return self.joint_positions.shape[1]

    @property
    def num_joints(self) -> int:
        return self.joint_positions.shape[2]

    @property
    def num_feet(self) -> int:
        return self.feet_frame_heights.shape[2]

    @property
    def body_weight(self) -> float:
        """Weight in newtons, the denominator of every force normalisation."""
        return self.body_mass * GRAVITY

    @property
    def sole_clearance(self) -> np.ndarray:
        """The best available clearance measure, sole where present, frame otherwise.

        The distinction is not cosmetic. The twenty first pass established that on a sole
        footed robot the frame height overestimates the true sole clearance by 23.5 mm on
        average and 74.4 mm at the 95th percentile, so a gate or a percentile computed on
        the frame is a different quantity. Where only the frame is available the caller is
        told so by `has_sole_clearance`, and every dependent record is labelled accordingly.

        CAUTION, recorded because the defect it guards against is deliberately left
        standing. play.py applies the SD_BRS1 sole table to whatever robot is played, the
        guard at play.py:299 testing a table literal rather than the configured global, so
        the PRESENCE of this channel is not evidence that the robot has a sole. Trust it
        on SD_BRS1 and read it as the frame height on anything else. See section 1.3.
        """
        if self.feet_sole_clearances is not None:
            return self.feet_sole_clearances
        return self.feet_frame_heights

    @property
    def has_sole_clearance(self) -> bool:
        return self.feet_sole_clearances is not None

    @property
    def has_peak_contact_force(self) -> bool:
        """Whether feet_contact_forces carries the history peak or the instantaneous force."""
        return self.contact_force_history_length > 0


def from_dump(dump: dict, rewards: dict, dt: float | None = None) -> GaitData:
    """Build a GaitData from a loaded dump.npy and its loaded rewards.npy.

    Args:
        dump: The dictionary saved as dump.npy, every per step key an array (T, E, ...).
        rewards: The dictionary saved as rewards.npy, every term an array (T, E) of
            RATES per second, carrying the logged weights under "_weights". MANDATORY.
            Every play writes this file beside dump.npy, so a caller that cannot supply
            it is holding an incomplete dump and must be told rather than accommodated.
        dt: Control period, overriding the one the dump records. Supply it from the run's
            own params/env.yaml through experiment_params.step_dt, since assuming 0.01 is
            wrong for half the tasks here.

    Raises:
        ValueError: If rewards is None, which means rewards.npy was missing or unread.
    """
    if rewards is None:
        raise ValueError(
            "rewards.npy is a mandatory member of a play dump and was not supplied. "
            "Every play writes it beside dump.npy, so its absence is a broken dump "
            "rather than an optional channel, and the reward budget statistics cannot "
            "be computed without it."
        )

    def series(key: str) -> np.ndarray | None:
        """A channel as stored, cast to float64, or None where the dump lacks it."""
        value = dump.get(key)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        return array if array.size else None

    def names(key: str) -> list[str]:
        return [str(n) for n in np.asarray(dump.get(key, [])).ravel()]

    masses = series("body_masses")
    if masses is None:
        # Fall back to the per step morphology channel, which every dump carries. Its
        # shape is (T, E, B), so the first step of the first environment is the row of
        # per body masses, and their sum is the articulation mass.
        per_step = series("robot_mass")
        masses = per_step[0, 0] if per_step is not None else None

    # Reserved keys are namespaced with a leading underscore and are not term series.
    weights = dict(rewards.get("_weights", {}))
    reward_series = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in rewards.items()
        if not k.startswith("_")
    }

    return GaitData(
        dt=float(dt if dt is not None else dump.get("step_dt", 0.01)),
        joint_names=names("joint_names"),
        body_names=names("body_names"),
        feet_names=names("feet_names"),
        joint_positions=series("joint_positions"),
        joint_velocities=series("joint_velocities"),
        joint_torques=series("joint_torques"),
        joint_accelerations=series("joint_accelerations"),
        joint_powers=series("joint_powers"),
        base_com_position=series("base_com_position"),
        base_linear_velocity=series("base_linear_velocity"),
        base_angular_velocity=series("base_angular_velocity"),
        commanded_linear_velocity=series("commanded_linear_velocity"),
        commanded_angular_velocity=series("commanded_angular_velocity"),
        feet_contact_forces=series("feet_contact_forces"),
        feet_velocities=series("feet_velocities"),
        feet_frame_heights=series("feet_frame_heights"),
        feet_sole_clearances=series("feet_sole_clearances"),
        feet_distance=series("feet_distance"),
        base_quaternion=series("base_quaternion"),
        base_projected_gravity=series("base_projected_gravity"),
        feet_quaternions=series("feet_quaternions"),
        gait_command=series("gait_command"),
        episode_dones=series("episode_dones"),
        episode_terminated=series("episode_terminated"),
        episode_time_outs=series("episode_time_outs"),
        joint_effort_limits=series("joint_effort_limits"),
        joint_velocity_limits=series("joint_velocity_limits"),
        joint_position_limits=series("joint_position_limits"),
        joint_soft_position_limits=series("joint_soft_position_limits"),
        default_joint_pos=series("default_joint_pos"),
        contact_force_history_length=int(dump.get("contact_force_history_length", 0)),
        frame_convention=dump.get("frame_convention"),
        body_mass=float(masses.sum()) if masses is not None else float("nan"),
        rewards=reward_series,
        reward_weights=weights,
    )


def record(group: str, quantity: str, statistic: str, value, unit: str = "") -> dict:
    """One cell of the output table."""
    return {
        "group": group,
        "quantity": quantity,
        "statistic": statistic,
        "value": float(value) if np.isscalar(value) or np.ndim(value) == 0 else np.nan,
        "unit": unit,
    }


def summarise(group: str, quantity: str, values, unit: str = "",
              percentiles=PERCENTILES, absolute: bool = False) -> list[dict]:
    """The uniform descriptive set applied to every scalar quantity.

    Mean, standard deviation, median, the six percentiles, the minimum, the maximum and
    the root mean square. The last is included because the record quotes it throughout for
    torque and for tracking error, and because for a zero mean oscillation it is the only
    one of the three central measures that does not vanish. `absolute` takes the magnitude
    first, which is the correct reading for a signed quantity whose sign alternates by
    construction, a joint torque or a foot velocity, where the mean of the signed series
    reports the drift and not the demand.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [record(group, quantity, "count", 0, "")]
    if absolute:
        values = np.abs(values)
    out = [
        record(group, quantity, "mean", np.mean(values), unit),
        record(group, quantity, "std", np.std(values), unit),
        record(group, quantity, "median", np.median(values), unit),
    ]
    for p in percentiles:
        out.append(record(group, quantity, f"p{p}", np.percentile(values, p), unit))
    out += [
        record(group, quantity, "min", np.min(values), unit),
        record(group, quantity, "max", np.max(values), unit),
        record(group, quantity, "rms", np.sqrt(np.mean(np.square(values))), unit),
        record(group, quantity, "count", values.size, ""),
    ]
    return out


def _correlation(a, b) -> float:
    """Pearson correlation, returning nan rather than raising on a degenerate input."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _yaw_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Yaw angle from a wxyz quaternion array of shape (..., 4)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def episode_segments(data: GaitData) -> list[tuple[int, int, int]]:
    """Contiguous (env, start, stop) segments free of any reset.

    A reset teleports the base, resamples the command and clears every contact
    accumulator, so a cycle period or a position error spanning one is meaningless.
    Where the dump carries `episode_dones` the boundaries are exact, which is every dump
    an environment with a termination manager produces. Where it does not, the getattr
    guard of section 4.2.3 having found no such manager, they are inferred from a base
    position discontinuity, a threshold of 0.2 m being far above the 0.03 m a single
    control step can produce at any plausible speed and far below the 19.5 m observed at
    a real reset.
    """
    T, E = data.num_steps, data.num_envs
    if data.episode_dones is not None:
        boundaries = np.asarray(data.episode_dones, dtype=bool)
    else:
        jump = np.linalg.norm(np.diff(data.base_com_position, axis=0), axis=-1)
        boundaries = np.zeros((T, E), dtype=bool)
        boundaries[1:] = jump > 0.2
    segments = []
    for e in range(E):
        edges = [0] + (np.flatnonzero(boundaries[:, e]) + 1).tolist() + [T]
        for start, stop in zip(edges[:-1], edges[1:]):
            if stop - start >= 2:
                segments.append((e, start, stop))
    return segments


def contact_state(data: GaitData, threshold: float = CONTACT_FORCE_THRESHOLD) -> np.ndarray:
    """Per foot contact indicator of shape (T, E, F)."""
    return np.linalg.norm(data.feet_contact_forces, axis=-1) > threshold


def touchdown_events(contact: np.ndarray) -> np.ndarray:
    """Indices (t, e, f) of every rising edge, t being the first step in contact."""
    rising = contact[1:] & ~contact[:-1]
    idx = np.argwhere(rising)
    idx[:, 0] += 1
    return idx


def swing_segments(contact: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Complete airborne intervals as (env, foot, start, stop), half open in stop.

    Only intervals bounded by a lift off and a touchdown are returned, so a foot airborne
    at the first or the last step of the dump contributes nothing, which is correct
    because its apex and its duration are both unobserved.
    """
    out = []
    T, E, F = contact.shape
    for e in range(E):
        for f in range(F):
            air = ~contact[:, e, f]
            starts = np.flatnonzero((~air[:-1]) & air[1:]) + 1
            stops = np.flatnonzero(air[:-1] & (~air[1:])) + 1
            for start in starts:
                later = stops[stops > start]
                if later.size:
                    out.append((e, f, int(start), int(later[0])))
    return out


def temporal_statistics(data: GaitData) -> list[dict]:
    """Cadence, duty factor, support phase occupancy and their variability.

    These are the coarsest description under which two gaits compare at all, and the
    record supplies the demonstration that they are not optional.
    """
    group = "temporal"
    contact = contact_state(data)
    n_in_contact = contact.sum(axis=-1)
    out = [
        record(group, "double support", "fraction",
               100.0 * np.mean(n_in_contact == 2), "pct"),
        record(group, "single support", "fraction",
               100.0 * np.mean(n_in_contact == 1), "pct"),
        record(group, "flight", "fraction",
               100.0 * np.mean(n_in_contact == 0), "pct"),
    ]
    # Duty factor per foot, the fraction of the cycle a given foot is loaded. A walk sits
    # above one half by construction and a run below it, so the value separates the two
    # gait classes without reference to speed.
    for f, name in enumerate(data.feet_names or range(data.num_feet)):
        out.append(record(group, f"stance duty {name}", "fraction",
                          100.0 * np.mean(contact[:, :, f]), "pct"))

    # Cycle period, taken between successive touchdowns of the SAME foot and restricted to
    # intervals free of a reset. The median and the mean must both be reported. The
    # twentieth pass records a median of exactly 1.000 s against a mean of 0.929, the
    # difference being a tail of short cycles whose tenth percentile lies at 0.720, which
    # is the signature of a stumble or of a contact breaking and re-forming within one
    # phase, and neither statistic alone reveals it.
    valid = np.zeros((data.num_steps, data.num_envs), dtype=bool)
    for e, start, stop in episode_segments(data):
        valid[start:stop, e] = True
    periods, stance_times, swing_times = [], [], []
    for e in range(data.num_envs):
        for f in range(data.num_feet):
            td = np.flatnonzero(contact[1:, e, f] & ~contact[:-1, e, f]) + 1
            td = td[valid[td, e]]
            if td.size > 1:
                gaps = np.diff(td) * data.dt
                # Drop any interval spanning a reset.
                keep = [
                    i for i, (a, b) in enumerate(zip(td[:-1], td[1:]))
                    if valid[a:b, e].all()
                ]
                periods.extend(gaps[keep].tolist())
    for e, f, start, stop in swing_segments(contact):
        if valid[start:stop, e].all():
            swing_times.append((stop - start) * data.dt)
    out += summarise(group, "cycle period", periods, "s")
    out += summarise(group, "swing duration", swing_times, "s")
    if periods:
        periods_arr = np.asarray(periods)
        out.append(record(group, "cadence", "steps per second",
                          2.0 / np.median(periods_arr) if np.median(periods_arr) else np.nan,
                          "1/s"))
        # Stride time variability. The clinical result grounding it is that this quantity
        # predicts falls where the mean stride time does not [16], and it answers a
        # question no percentile answers, whether the gait is a limit cycle or a sequence
        # of recoveries.
        out.append(record(group, "cycle period", "coefficient of variation",
                          100.0 * np.std(periods_arr) / np.mean(periods_arr), "pct"))

    # Commanded schedule, where a gait clock exists. The identity is exact for an anti
    # phase offset of one half, the stance duty per foot equalling the duration and the
    # double support fraction equalling 2 * duration - 1, which the twenty first pass
    # established and which corrects the 19.6 percent of the twentieth pass to 20.0.
    if data.gait_command is not None and data.gait_command.shape[-1] >= 3:
        duration = float(np.mean(data.gait_command[..., 2]))
        offset = float(np.mean(data.gait_command[..., 1]))
        out.append(record(group, "commanded stance duty", "value", 100.0 * duration, "pct"))
        if abs(offset - 0.5) < 1e-6:
            out.append(record(group, "commanded double support", "value",
                              100.0 * (2.0 * duration - 1.0), "pct"))
        out.append(record(group, "commanded frequency", "value",
                          float(np.mean(data.gait_command[..., 0])), "Hz"))
    return out


def contact_event_statistics(data: GaitData) -> list[dict]:
    """Touchdown velocities, peak forces, loading rates and impulses.

    Two velocity measures must both be reported and must not be confused. The frame
    vertical velocity is what the superseded foot_landing_vel charged, and the sole
    approach velocity is what the collision is actually governed by, the two differing by
    the rotational term.

    The force channel is the per step maximum over the contact sensor's history, which is
    the quantity the impact reward prices, wherever the sensor declares a history. Where
    it declares none the sensor falls back to the instantaneous force and the channel
    reduces to what it was before section 4.2.5. The two are not interchangeable, the
    latter understating the former by a factor the twenty first pass measured at twelve,
    so the provenance is reported beside the figures through the history depth rather than
    left to be inferred.
    """
    group = "impact"
    contact = contact_state(data)
    events = touchdown_events(contact)
    window = max(1, int(round(IMPACT_WINDOW_S / data.dt)))
    force_norm = np.linalg.norm(data.feet_contact_forces, axis=-1)
    clearance = data.sole_clearance
    descent = -(clearance[1:] - clearance[:-1]) / data.dt   # positive downward

    frame_vz, sole_vz, peaks, rates, impulses = [], [], [], [], []
    for t, e, f in events:
        frame_vz.append(abs(data.feet_velocities[t - 1, e, f, 2]))
        sole_vz.append(descent[t - 1, e, f])
        segment = force_norm[t:t + window, e, f]
        if segment.size:
            peaks.append(segment.max())
            # Loading rate, the rise of the force through the first hump divided by the
            # time it took. A collision and a weight acceptance may reach the same peak
            # and are told apart only here, the record measuring 733 body weights per
            # second for a gait whose peak alone read merely high.
            rise = int(np.argmax(segment))
            if rise > 0:
                rates.append(segment.max() / (rise * data.dt))
        # Impulse over the following stance, the time integral of the normal force, which
        # must equal the momentum change the step requires and therefore serves as a
        # conservation check on the contact model. CAVEAT, from section 4.2.5 onward the
        # summand is the per step MAXIMUM over the force history rather than a sample of
        # it, so this over-reads the true impulse wherever the force varies within a
        # control step, most of all through the collision itself. It remains valid as a
        # comparison between two runs measured the same way and as a bound above, and it
        # is not a substitute for the momentum balance a dedicated instantaneous channel
        # would permit. Recorded rather than repaired, since restoring the instantaneous
        # force would mean a second channel of the same size for one statistic.
        stance_end = t
        while stance_end < contact.shape[0] and contact[stance_end, e, f]:
            stance_end += 1
        impulses.append(force_norm[t:stance_end, e, f].sum() * data.dt)

    bw = data.body_weight
    out = summarise(group, "touchdown frame vertical speed", frame_vz, "m/s")
    out += summarise(group, "touchdown sole approach speed", sole_vz, "m/s")
    out += summarise(group, "peak contact force", np.asarray(peaks) / bw, "BW")
    out += summarise(group, "loading rate", np.asarray(rates) / bw, "BW/s")
    out += summarise(group, "stance impulse", np.asarray(impulses) / bw, "BW·s")
    if frame_vz and sole_vz:
        # The ratio itself, since it is the signature of the rotational substitution.
        out.append(record(group, "sole to frame speed ratio", "mean",
                          np.mean(sole_vz) / np.mean(np.abs(frame_vz)), ""))
    out.append(record(group, "touchdowns", "count", len(events), ""))
    # The provenance of every force figure above, stated as a record so that it travels
    # with the numbers into the comparison table rather than living in a footnote.
    out.append(record(group, "contact force history depth", "steps",
                      data.contact_force_history_length, "physics steps"))
    return out


def swing_profile_statistics(data: GaitData, samples: int = 21) -> list[dict]:
    """Apex, apex position, peak count and correlation against a raised cosine."""
    group = "swing"
    contact = contact_state(data)
    clearance = data.sole_clearance
    phi = np.linspace(0.0, 1.0, samples)
    reference = np.sin(np.pi * phi) ** 2

    apex, apex_pos, peaks, corr, profiles = [], [], [], [], []
    for e, f, start, stop in swing_segments(contact):
        profile = clearance[start:stop, e, f]
        if profile.size < 6:
            continue
        apex.append(profile.max())
        apex_pos.append(np.argmax(profile) / (profile.size - 1))
        interior = profile[1:-1]
        n_peaks = int(((interior > profile[:-2]) & (interior >= profile[2:])).sum())
        peaks.append(max(n_peaks, 1))
        resampled = np.interp(phi, np.linspace(0.0, 1.0, profile.size), profile)
        profiles.append(resampled)
        corr.append(_correlation(resampled, reference))

    out = summarise(group, "swing apex clearance", apex, "m")
    out += summarise(group, "swing apex position", apex_pos, "fraction")
    out += summarise(group, "clearance peaks per swing", peaks, "count")
    out += summarise(group, "raised cosine correlation", corr, "")
    if profiles:
        mean_profile = np.mean(profiles, axis=0)
        # The plateau residue a set point kernel cannot remove, which is what sizes the
        # width of a reference tracking kernel. The twenty second pass measures the foot
        # spending 65.6 percent of its swing within one standard deviation of the set
        # point where the reference spends 46.0.
        out.append(record(group, "swing profile", "fraction above 90 pct of apex",
                          100.0 * np.mean(mean_profile >= 0.9 * mean_profile.max()), "pct"))
        scale = mean_profile.max() if mean_profile.max() > 0 else 1.0
        out.append(record(group, "swing profile", "rms error against raised cosine",
                          float(np.sqrt(np.mean((mean_profile - scale * reference) ** 2))),
                          "m"))
    out.append(record(group, "clearance source",
                      "is true sole" if data.has_sole_clearance else "is frame proxy",
                      1, ""))
    return out


def joint_statistics(data: GaitData) -> list[dict]:
    """Per joint and aggregate moments, percentiles, peaks and margins.

    Reported both per joint and in aggregate, because the two answer different questions.
    The aggregate says what the machine costs and the per joint says which actuator pays.
    The percentiles are taken on the MAGNITUDE, because every one of these
    quantities alternates in sign by construction, so the mean of the signed series
    reports the drift and the mean of the magnitude reports the demand.
    """
    group = "joints"
    channels = (
        ("joint velocity", data.joint_velocities, "rad/s", data.joint_velocity_limits),
        ("joint torque", data.joint_torques, "N·m", data.joint_effort_limits),
        ("joint power", data.joint_powers, "W", None),
        ("joint acceleration", data.joint_accelerations, "rad/s^2", None),
        ("joint position", data.joint_positions, "rad", None),
    )
    names = data.joint_names or [f"joint {j}" for j in range(data.num_joints)]
    out = []
    for label, series, unit, ceiling in channels:
        if series is None:
            continue
        # Aggregate over every joint and every step.
        out += summarise(group, f"{label}, all joints", series, unit, absolute=True)
        # The peak of the per step sum, which is the instantaneous whole body demand and
        # is not recoverable from the per joint percentiles, those being taken
        # independently and therefore never coincident.
        out += summarise(group, f"{label}, whole body sum",
                         np.abs(series).sum(axis=-1), unit)
        for j, name in enumerate(names):
            column = series[:, :, j]
            out.append(record(group, f"{label} {name}", "mean absolute",
                              np.mean(np.abs(column)), unit))
            out.append(record(group, f"{label} {name}", "std", np.std(column), unit))
            out.append(record(group, f"{label} {name}", "median absolute",
                              np.median(np.abs(column)), unit))
            for p in PERCENTILES:
                out.append(record(group, f"{label} {name}", f"p{p} absolute",
                                  np.percentile(np.abs(column), p), unit))
            out.append(record(group, f"{label} {name}", "peak", np.max(np.abs(column)), unit))
            out.append(record(group, f"{label} {name}", "range",
                              np.max(column) - np.min(column), unit))
            if ceiling is not None and j < len(ceiling) and ceiling[j] > 0:
                # Saturation. An absolute penalty cannot see this, since the same torque
                # is unremarkable at a hip and saturating at an ankle, which is the
                # argument for Booster Gym's torque tiredness term [19].
                out.append(record(
                    group, f"{label} {name}", "p99 as fraction of ceiling",
                    100.0 * np.percentile(np.abs(column), 99) / ceiling[j], "pct"))
                out.append(record(
                    group, f"{label} {name}", "fraction above 98 pct of ceiling",
                    100.0 * np.mean(np.abs(column) > EFFORT_SATURATION_FRACTION * ceiling[j]),
                    "pct"))
                out.append(record(
                    group, f"{label} {name}", "fraction exceeding ceiling",
                    100.0 * np.mean(np.abs(column) > ceiling[j]), "pct"))

    # Limit residence. A policy that discovers a mechanical stop is a cheap source of
    # support will lean on it, and the twentieth pass finds the knee within 0.02 rad of a
    # limit for 37 percent of steps with a distribution so bimodal that only 15 percent
    # falls in the intermediate band. The stop is not a joint being controlled, it is a
    # joint being switched between two constraints.
    if data.joint_position_limits is not None:
        lower, upper = data.joint_position_limits[:, 0], data.joint_position_limits[:, 1]
        for j, name in enumerate(names):
            column = data.joint_positions[:, :, j]
            at_limit = (
                (column - lower[j] < LIMIT_TOLERANCE_RAD)
                | (upper[j] - column < LIMIT_TOLERANCE_RAD)
            )
            out.append(record(group, f"joint position {name}", "fraction at limit",
                              100.0 * np.mean(at_limit), "pct"))
            out.append(record(group, f"joint position {name}", "fraction beyond limit",
                              100.0 * np.mean((column < lower[j]) | (column > upper[j])),
                              "pct"))
    return out


def foot_statistics(data: GaitData) -> list[dict]:
    """Per foot forces, velocities, clearances and orientations, in stance and in swing.

    Every quantity is reported three ways, over all steps, over the steps in contact and
    over the steps airborne, because the aggregate conceals the distinction that matters.
    A horizontal foot speed of 3.40 m/s at the 99th percentile is a swing property and a
    horizontal foot speed of 0.158 m/s is a stance property, and the first is the tanh
    factor of the clearance reward being farmed while the second is the slide penalty.
    """
    group = "feet"
    contact = contact_state(data)
    names = data.feet_names or [f"foot {f}" for f in range(data.num_feet)]
    force = data.feet_contact_forces
    velocity = data.feet_velocities
    bw = data.body_weight
    out = []
    for f, name in enumerate(names):
        in_contact = contact[:, :, f]
        airborne = ~in_contact
        channels = (
            ("contact force magnitude", np.linalg.norm(force[:, :, f], axis=-1) / bw, "BW"),
            ("contact force vertical", force[:, :, f, 2] / bw, "BW"),
            ("contact force horizontal",
             np.linalg.norm(force[:, :, f, :2], axis=-1) / bw, "BW"),
            ("contact force x", force[:, :, f, 0] / bw, "BW"),
            ("contact force y", force[:, :, f, 1] / bw, "BW"),
            ("foot speed horizontal",
             np.linalg.norm(velocity[:, :, f, :2], axis=-1), "m/s"),
            ("foot velocity vertical", velocity[:, :, f, 2], "m/s"),
            ("foot velocity x", velocity[:, :, f, 0], "m/s"),
            ("foot velocity y", velocity[:, :, f, 1], "m/s"),
            ("frame height", data.feet_frame_heights[:, :, f], "m"),
            ("sole clearance", data.sole_clearance[:, :, f], "m"),
        )
        for label, series, unit in channels:
            out += summarise(group, f"{label} {name}", series, unit)
            out += summarise(group, f"{label} {name}, in stance", series[in_contact], unit)
            out += summarise(group, f"{label} {name}, in swing", series[airborne], unit)
        # Slip, the horizontal distance a loaded foot travels, which is the quantity
        # feet_slide prices and which is also the stance pivot a flat footed biped needs
        # in order to turn, so it must not be driven to zero.
        slip = np.linalg.norm(velocity[:, :, f, :2], axis=-1) * in_contact * data.dt
        out.append(record(group, f"stance slip distance {name}", "mean per stance step",
                          np.sum(slip) / max(in_contact.sum(), 1), "m"))
        # The frame to sole discrepancy, which is the tilt the frame height cannot
        # distinguish from a lift, and which the twenty first pass measured at 23.5 mm.
        if data.has_sole_clearance:
            out += summarise(group, f"frame minus sole clearance {name}",
                             data.feet_frame_heights[:, :, f]
                             - data.feet_sole_clearances[:, :, f], "m")
        # Forged contact, a foot reporting force while its sole is demonstrably clear of
        # the ground. This is the ninth pass signature, and it was caused by a shank into
        # foot interpenetration in the mesh asset which silenced every contact keyed term
        # in the reward set for eight passes before it was found.
        if data.has_sole_clearance:
            forged = in_contact & (data.feet_sole_clearances[:, :, f] > 0.01)
            out.append(record(group, f"forged contact {name}", "fraction",
                              100.0 * np.mean(forged), "pct"))
        if data.feet_quaternions is not None:
            quat = data.feet_quaternions[:, :, f]
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
            out += summarise(group, f"foot roll {name}", roll, "rad", absolute=True)
            out += summarise(group, f"foot pitch {name}", pitch, "rad", absolute=True)
            out += summarise(group, f"foot roll {name}, in stance", roll[in_contact],
                             "rad", absolute=True)
            if data.base_quaternion is not None:
                # Foot yaw alignment, the term Booster Gym carries at twice its linear
                # tracking weight and which is absent from the SD_BRS1 set [19].
                misalignment = np.arctan2(
                    np.sin(_yaw_from_quaternion(quat) - _yaw_from_quaternion(data.base_quaternion)),
                    np.cos(_yaw_from_quaternion(quat) - _yaw_from_quaternion(data.base_quaternion)),
                )
                out += summarise(group, f"foot yaw misalignment {name}", misalignment,
                                 "rad", absolute=True)
    return out


def feet_distance_statistics(data: GaitData) -> list[dict]:
    """Signed per axis separation, its norms, and the correlations among them.

    The decisive statistic here is the LATERAL component alone and not the planar norm,
    since a hinge on the planar separation is satisfied by a long stride whatever the
    lateral separation happens to be. Any configuration adopting such a term must confirm
    which axis it measures before trusting it to regulate stance width.

    THE COMPONENTS ARE ONLY MEANINGFUL IN THE BASE FRAME, and the dump must supply them
    there. A world frame separation is the same VECTOR and has the same magnitude, the
    planar norms agreeing to float32 identity, but its horizontal components are a
    rotating mixture of the stance width and the stride whenever the heading is not zero.
    Since 2026-08-07 play.py rotates the separation by the base yaw before logging it and
    records the fact under the frame_convention key. A dump lacking that key predates the
    change and its horizontal components are world frame, in which case this function's
    fore and aft and lateral records must not be read.

    The correction withdrew a finding this docstring previously asserted. The twentieth
    and twenty second passes reported a lateral mean of 0.206 m with a fifth percentile
    near zero and concluded that the feet crossed the midline. Both figures were world
    frame. The base frame reading of the same run gives a mean of 0.2591 m, a fifth
    percentile of 0.2169, a minimum of 0.0907 and no sign change in 96032 samples, so the
    feet do not cross and the stance is narrow rather than collapsing. See section 12.2 of
    /ws/context/gait_metrics.md and the correction appended to section 3.6.7 of
    /ws/plans/GAIT_EFFICIENCY_PLAN.md.
    """
    group = "stance"
    if data.feet_distance is None:
        return []
    separation = data.feet_distance
    fore_aft, lateral, vertical = separation[..., 0], separation[..., 1], separation[..., 2]
    planar = np.linalg.norm(separation[..., :2], axis=-1)
    total = np.linalg.norm(separation, axis=-1)
    out = []
    for label, series in (
        ("fore and aft separation", fore_aft),
        ("lateral separation", lateral),
        ("vertical separation", vertical),
    ):
        out += summarise(group, label, series, "m")
        out += summarise(group, f"{label} magnitude", series, "m", absolute=True)
    out += summarise(group, "planar separation", planar, "m")
    out += summarise(group, "total separation", total, "m")
    # The pairwise correlations, which say whether the axes trade against one another.
    # A strongly negative correlation between the lateral and the fore and aft magnitudes
    # is the substitution above, made visible without any threshold being assumed.
    pairs = (
        ("fore and aft against lateral", fore_aft, lateral),
        ("fore and aft against vertical", fore_aft, vertical),
        ("lateral against vertical", lateral, vertical),
        ("planar against vertical", planar, vertical),
        ("lateral magnitude against fore and aft magnitude",
         np.abs(lateral), np.abs(fore_aft)),
    )
    for label, a, b in pairs:
        out.append(record(group, f"separation correlation, {label}", "pearson",
                          _correlation(a, b), ""))
    return out


def tracking_statistics(data: GaitData) -> list[dict]:
    """Velocity tracking errors, correlations, and the components that should vanish.

    Six components are reported. The three commanded ones carry an error and a
    correlation, and the three uncommanded ones, the lateral linear velocity and the roll
    and pitch rates, carry a magnitude, since for those the ideal is zero and an error
    against a zero command is the magnitude itself.

    Two cautions from the record govern the reading. The yaw command is a heading
    controller output rather than a sample where `heading_command` is true, so it saturates
    at the range limit while the error persists and a robot that cannot turn accumulates a
    command it still cannot follow. And the tracking error does not converge downward while
    a curriculum is active, the reward kernel contracting and the command range widening as
    training proceeds, so runs must be compared at matched iteration counts.
    """
    group = "tracking"
    measured_lin = data.base_linear_velocity
    measured_ang = data.base_angular_velocity
    commanded_lin = data.commanded_linear_velocity
    commanded_ang = np.asarray(data.commanded_angular_velocity)
    if commanded_ang.ndim == 3:
        commanded_ang = commanded_ang[..., 0]

    out = []
    axes = (("x", 0), ("y", 1))
    for label, axis in axes:
        error = measured_lin[..., axis] - commanded_lin[..., axis]
        out += summarise(group, f"linear velocity error {label}", error, "m/s")
        out += summarise(group, f"linear velocity error {label} magnitude", error,
                         "m/s", absolute=True)
        out.append(record(group, f"linear velocity {label}", "command correlation",
                          _correlation(measured_lin[..., axis], commanded_lin[..., axis]), ""))
        out.append(record(group, f"linear velocity {label}", "commanded magnitude mean",
                          np.mean(np.abs(commanded_lin[..., axis])), "m/s"))
        out.append(record(group, f"linear velocity {label}", "achieved magnitude mean",
                          np.mean(np.abs(measured_lin[..., axis])), "m/s"))
    planar_error = np.linalg.norm(measured_lin[..., :2] - commanded_lin[..., :2], axis=-1)
    out += summarise(group, "planar velocity error", planar_error, "m/s")

    yaw_error = measured_ang[..., 2] - commanded_ang
    out += summarise(group, "yaw rate error", yaw_error, "rad/s")
    out.append(record(group, "yaw rate", "command correlation",
                      _correlation(measured_ang[..., 2], commanded_ang), ""))
    out.append(record(group, "yaw rate", "commanded magnitude mean",
                      np.mean(np.abs(commanded_ang)), "rad/s"))
    out.append(record(group, "yaw rate", "achieved magnitude mean",
                      np.mean(np.abs(measured_ang[..., 2])), "rad/s"))
    # The three that should be zero. Reporting them as magnitudes rather than as errors
    # is deliberate, since an error against a zero command IS the magnitude and naming it
    # an error invites a reader to look for a command that does not exist.
    for label, series, unit in (
        ("lateral linear velocity", measured_lin[..., 1], "m/s"),
        ("vertical linear velocity", measured_lin[..., 2], "m/s"),
        ("roll rate", measured_ang[..., 0], "rad/s"),
        ("pitch rate", measured_ang[..., 1], "rad/s"),
    ):
        out += summarise(group, f"{label} magnitude", series, unit, absolute=True)
    return out


def odometry_statistics(data: GaitData) -> list[dict]:
    """Error between the base position a commanded trajectory implies and the measured one.

    The integration is performed within an episode segment and reset at every boundary,
    because a reset teleports the base and an error accumulated across one is unbounded.
    The error is reported both in absolute metres and normalised by the distance the
    command asked for, the second being the comparable figure across runs whose command
    distributions differ.
    """
    group = "odometry"
    if data.base_quaternion is None:
        return [record(group, "expected base position error",
                       "requires base_quaternion, absent", np.nan, "m")]
    yaw = _yaw_from_quaternion(data.base_quaternion)
    errors, lateral_errors, along_errors, commanded_distance = [], [], [], []
    for e, start, stop in episode_segments(data):
        cos, sin = np.cos(yaw[start:stop, e]), np.sin(yaw[start:stop, e])
        vx, vy = data.commanded_linear_velocity[start:stop, e, 0], \
                 data.commanded_linear_velocity[start:stop, e, 1]
        # Rotate the body frame command into the world frame at the measured heading.
        world_vx, world_vy = cos * vx - sin * vy, sin * vx + cos * vy
        expected = np.cumsum(np.stack([world_vx, world_vy], axis=-1), axis=0) * data.dt
        measured = (data.base_com_position[start:stop, e, :2]
                    - data.base_com_position[start, e, :2])
        delta = measured - expected
        errors.extend(np.linalg.norm(delta, axis=-1).tolist())
        # Decomposed along and across the commanded heading, since a lag along the
        # direction of travel is a speed deficit while a drift across it is a heading
        # failure, and the two call for different remedies.
        norm = np.linalg.norm(expected, axis=-1)
        norm[norm == 0] = 1.0
        unit = expected / norm[:, None]
        along_errors.extend((delta * unit).sum(axis=-1).tolist())
        lateral_errors.extend((delta[:, 0] * -unit[:, 1] + delta[:, 1] * unit[:, 0]).tolist())
        commanded_distance.append(float(np.linalg.norm(expected[-1])))
    out = summarise(group, "expected base position error", errors, "m")
    out += summarise(group, "position error along command", along_errors, "m")
    out += summarise(group, "position error across command", lateral_errors, "m")
    if commanded_distance and np.sum(commanded_distance) > 0:
        out.append(record(group, "expected base position error",
                          "as fraction of commanded distance",
                          100.0 * np.mean(errors) / np.mean(commanded_distance), "pct"))
    out += summarise(group, "base height", data.base_com_position[..., 2], "m")
    return out


def energetic_statistics(data: GaitData) -> list[dict]:
    """Power, cost of transport in three conventions, and the antagonism ratio.

    The three conventions diverge by an order of magnitude on a badly conditioned gait and
    must all be reported.
    """
    group = "energetics"
    power = data.joint_powers
    absolute = np.abs(power).sum(axis=-1)
    net = power.sum(axis=-1)
    positive = np.clip(net, 0.0, None)
    speed = float(np.mean(np.linalg.norm(data.base_linear_velocity[..., :2], axis=-1)))
    denominator = data.body_weight * speed if speed > 0 else np.nan

    out = summarise(group, "absolute joint power", absolute, "W")
    out += summarise(group, "net joint power", net, "W")
    out.append(record(group, "cost of transport, absolute", "value",
                      np.mean(absolute) / denominator, ""))
    out.append(record(group, "cost of transport, net", "value",
                      np.mean(np.abs(net)) / denominator, ""))
    out.append(record(group, "cost of transport, positive", "value",
                      np.mean(positive) / denominator, ""))
    out.append(record(group, "antagonism ratio", "absolute over net",
                      np.mean(absolute) / max(np.mean(np.abs(net)), 1e-9), ""))
    out.append(record(group, "mean forward speed", "value", speed, "m/s"))
    # Pairwise opposition, which localises the antagonism to a joint pair. The twenty
    # second pass reports the hip pitch and the knee expending 170 W against one another
    # falling to 43, which is the clearest single statement of what Phase 1b bought, and
    # it is invisible in any aggregate.
    names = data.joint_names or [f"joint {j}" for j in range(data.num_joints)]
    for a in range(data.num_joints):
        for b in range(a + 1, data.num_joints):
            pa, pb = power[:, :, a], power[:, :, b]
            opposed = np.sign(pa) != np.sign(pb)
            if opposed.mean() < 0.4:
                continue
            out.append(record(group, f"opposed power, {names[a]} against {names[b]}",
                              "mean", np.mean(np.minimum(np.abs(pa), np.abs(pb)) * opposed),
                              "W"))
            out.append(record(group, f"opposed power, {names[a]} against {names[b]}",
                              "fraction of steps opposed", 100.0 * opposed.mean(), "pct"))
    # Dimensionless speed, which is what makes a cadence comparable across leg lengths.
    # The Froude number is v^2 / (g L), and human walking transitions to running near 0.5.
    leg_length = float(np.mean(data.base_com_position[..., 2]))
    if leg_length > 0:
        out.append(record(group, "Froude number", "value",
                          speed ** 2 / (GRAVITY * leg_length), ""))
    return out


def posture_statistics(data: GaitData) -> list[dict]:
    """Torso attitude, and the three stability margins the literature prescribes."""
    group = "posture"
    out = []
    if data.base_projected_gravity is not None:
        # The xy norm of the projected gravity is the sine of the tilt from vertical, and
        # is exactly the quantity flat_orientation_l2 squares.
        tilt = np.arcsin(np.clip(
            np.linalg.norm(data.base_projected_gravity[..., :2], axis=-1), 0.0, 1.0))
        out += summarise(group, "torso tilt from vertical", np.degrees(tilt), "deg")
        out += summarise(group, "projected gravity xy norm",
                         np.linalg.norm(data.base_projected_gravity[..., :2], axis=-1), "")
    if data.base_quaternion is not None:
        quat = data.base_quaternion
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        out += summarise(group, "torso roll", np.degrees(roll), "deg", absolute=True)
        out += summarise(group, "torso pitch", np.degrees(pitch), "deg", absolute=True)
    out += summarise(group, "base height", data.base_com_position[..., 2], "m")
    out += summarise(group, "base vertical velocity", data.base_linear_velocity[..., 2],
                     "m/s", absolute=True)
    # Vertical centre of mass excursion across a stride, the human figure being about
    # five centimetres, which is the pendular exchange of potential against kinetic energy.
    excursions = []
    for e, start, stop in episode_segments(data):
        column = data.base_com_position[start:stop, e, 2]
        if column.size > 10:
            excursions.append(column.max() - column.min())
    out += summarise(group, "base height excursion per episode", excursions, "m")
    return out


def stability_statistics(data: GaitData) -> list[dict]:
    """Margin of stability, capture point excursion, and centre of pressure.

    The margin of stability of Hof, Gazendam and Sinke is the signed distance from the
    extrapolated centre of mass, being the position plus the velocity divided by the
    pendulum eigenfrequency, to the boundary of the base of support [8]. Its mediolateral
    component is the one associated with falls, and the frontal plane result that grounds
    it is that the lateral excursion required per step is half the stance width, so a
    narrow stance leaves nothing in reserve [9]. A negative value means a step is required
    to avoid a fall, which is normal during walking and pathological during standing.

    The base of support is approximated by the convex hull of the loaded feet positions,
    which for a two footed robot in single stance degenerates to a point and in double
    stance to a segment. That approximation understates the true polygon by the sole
    dimensions, which are not recorded in the dump, so the margin computed here is a lower
    bound and is labelled as such. A future extension carrying the sole offsets would
    close it.
    """
    group = "stability"
    contact = contact_state(data)
    height = np.mean(data.base_com_position[..., 2])
    omega = np.sqrt(GRAVITY / height) if height > 0 else np.nan

    # CHANGED 2026-08-07, the whole construction is now carried out in the BASE frame and
    # the base position no longer enters it. The margin is a difference of two points,
    # the extrapolated centre of mass and the support centroid, and both are expressed
    # relative to the same base position, so that position cancels identically. Writing
    # the cancellation out rather than performing it numerically removes the yaw rotation
    # this function previously needed, removes the dependence on base_quaternion with it,
    # and removes the possibility of mixing frames, which is the defect this replaces.
    #
    # The defect was that the components of the result were labelled fore and aft and
    # lateral while being computed in the world frame, so both were a rotating mixture of
    # the two whenever the heading was not zero, exactly as in feet_distance_statistics.
    # Since 2026-08-07 the dump supplies feet_distance already in the base frame, which
    # would have made the previous form worse rather than better, the world frame centre
    # of mass then being combined with a base frame separation.
    #
    #   xcom     = com + v / omega
    #   support  = com                (double stance, the midpoint of the two feet)
    #            = com + half         (foot 0 loaded)
    #            = com - half         (foot 1 loaded)
    #   delta    = xcom - support = v / omega  -/+ half
    #
    # base_linear_velocity is already the base frame velocity at play.py's log, and half
    # is now the base frame half separation, so delta is base frame throughout with no
    # rotation performed anywhere.
    velocity = data.base_linear_velocity[..., :2] / omega

    # Feet positions are not logged directly, but the signed separation and the contact
    # state together locate the support. Where the separation is absent the margin is
    # reported against the loaded foot count alone.
    margins_lateral, margins_fore_aft = [], []
    # The construction combines a BASE frame velocity with the recorded separation, so
    # the separation must be base frame too. A dump written before 2026-08-07 carries it
    # in the world frame, and mixing the two would produce a plausible number that is
    # meaningless, so such a dump is refused here rather than served. The same guard
    # governs the centre of pressure offset below, which reads the same half separation.
    separation_is_base = (data.frame_convention or {}).get("feet_distance") == "base_yaw"
    if data.feet_distance is not None and separation_is_base:
        # Half the recorded separation, which is foot0 minus foot1, so adding it to the
        # base moves toward foot0 and subtracting it moves toward foot1.
        half = data.feet_distance[..., :2] * 0.5
        loaded = contact.sum(axis=-1)
        offset = np.where(
            (loaded == 2)[..., None], 0.0,
            np.where(contact[..., 0:1], half, -half),
        )
        delta = velocity - offset
        margins_fore_aft.extend(delta[..., 0].ravel().tolist())
        margins_lateral.extend(delta[..., 1].ravel().tolist())
    out = summarise(group, "extrapolated com offset, fore and aft", margins_fore_aft, "m")
    out += summarise(group, "extrapolated com offset, lateral", margins_lateral, "m")
    out.append(record(group, "pendulum eigenfrequency", "value", omega, "1/s"))
    # Centre of pressure under the loaded feet, the observable half of the zero moment
    # point criterion [4], computed as the force weighted mean of the loaded foot
    # positions. Its excursion measures how near the foot is to rolling onto an edge.
    force_z = np.clip(data.feet_contact_forces[..., 2], 0.0, None)
    total = force_z.sum(axis=-1)
    if data.feet_distance is not None and separation_is_base:
        weights = np.divide(force_z, total[..., None], out=np.zeros_like(force_z),
                            where=total[..., None] > 0)
        cop_offset = (weights[..., 0] - weights[..., 1])[..., None] * half
        out += summarise(group, "centre of pressure offset, lateral",
                         cop_offset[..., 1], "m")
        out += summarise(group, "centre of pressure offset, fore and aft",
                         cop_offset[..., 0], "m")
    out.append(record(group, "base of support", "is a lower bound, sole extent unknown",
                      1, ""))
    return out


def _robinson_index(left: float, right: float) -> float:
    """Symmetry index of Robinson, Herzog and Nigg [12], as a percentage.

    Unbounded as the mean approaches zero, which is why it is applied here to magnitudes
    such as a root mean square torque or a range of motion rather than to a signed series.
    """
    denominator = 0.5 * (abs(left) + abs(right))
    if denominator == 0:
        return float("nan")
    return 100.0 * (left - right) / denominator


def symmetry_statistics(data: GaitData) -> list[dict]:
    """Left against right, on the quantities the record has found discriminating.

    Pairing is by name, a joint whose name differs from another's only in a trailing L or
    R being taken as its partner, which generalises across robots without a hardcoded map
    and degrades to reporting nothing where the naming convention differs. The record's
    use of these indices is instructive, the sagittal figures in single digits being cited
    as evidence the symmetry augmentation works while the frontal figure of 41 percent is
    cited as evidence of a lateral balance defect, so the value of the family lies in
    localising an asymmetry rather than in scoring it.
    """
    group = "symmetry"
    names = data.joint_names or []
    pairs = []
    for i, name in enumerate(names):
        if name.endswith("L"):
            partner = name[:-1] + "R"
            if partner in names:
                pairs.append((name[:-1], i, names.index(partner)))
    out = []
    for label, li, ri in pairs:
        for quantity, series in (
            ("torque", data.joint_torques),
            ("velocity", data.joint_velocities),
            ("power", data.joint_powers),
        ):
            # PER ENVIRONMENT, then averaged as a magnitude. Pooling the environments
            # before taking the index cancels the asymmetry, because the left and the
            # right pooled statistics each take the union over environments whose
            # asymmetries point in opposite directions. Measured on run
            # 2026-07-28_06-37-24, the ankle roll range of motion index reads 4.68
            # percent pooled, minus 5.17 as a signed per environment mean, and 13.4 as a
            # mean magnitude, against 14.09 in environment zero alone which is the panel
            # the plots display. The pooled figure is the one that misleads.
            per_env = [
                _robinson_index(
                    float(np.sqrt(np.mean(series[:, e, li] ** 2))),
                    float(np.sqrt(np.mean(series[:, e, ri] ** 2))),
                )
                for e in range(data.num_envs)
            ]
            per_env = np.asarray(per_env, dtype=np.float64)
            per_env = per_env[np.isfinite(per_env)]
            if per_env.size:
                out.append(record(group, f"{quantity} rms symmetry index {label}",
                                  "mean magnitude", np.mean(np.abs(per_env)), "pct"))
                out.append(record(group, f"{quantity} rms symmetry index {label}",
                                  "signed mean", np.mean(per_env), "pct"))
                out.append(record(group, f"{quantity} rms symmetry index {label}",
                                  "p99 magnitude", np.percentile(np.abs(per_env), 99), "pct"))
        # np.ptp rather than the ndarray method, the latter having been removed in
        # numpy 2.0 and the analysis environment carrying that version.
        rom = np.asarray([
            _robinson_index(float(np.ptp(data.joint_positions[:, e, li])),
                            float(np.ptp(data.joint_positions[:, e, ri])))
            for e in range(data.num_envs)
        ], dtype=np.float64)
        rom = rom[np.isfinite(rom)]
        if rom.size:
            out.append(record(group, f"range of motion symmetry index {label}",
                              "mean magnitude", np.mean(np.abs(rom)), "pct"))
            out.append(record(group, f"range of motion symmetry index {label}",
                              "signed mean", np.mean(rom), "pct"))
    # Per foot temporal and kinetic symmetry, which is where a limp shows first.
    contact = contact_state(data)
    if data.num_feet == 2:
        duties = [float(np.mean(contact[:, :, f])) for f in range(2)]
        out.append(record(group, "stance duty symmetry index", "value",
                          _robinson_index(*duties), "pct"))
        swings = [[] for _ in range(2)]
        for e, f, start, stop in swing_segments(contact):
            swings[f].append((stop - start) * data.dt)
        if all(swings):
            out.append(record(group, "swing duration symmetry index", "value",
                              _robinson_index(np.mean(swings[0]), np.mean(swings[1])), "pct"))
        peaks = [float(np.percentile(
            np.linalg.norm(data.feet_contact_forces[:, :, f], axis=-1), 99))
            for f in range(2)]
        out.append(record(group, "peak contact force symmetry index", "value",
                          _robinson_index(*peaks), "pct"))
    return out


def _spectral_arc_length(signal: np.ndarray, dt: float,
                         cutoff: float = 20.0, amplitude_threshold: float = 0.05) -> float:
    """Spectral arc length of Balasubramanian and colleagues [14].

    The arc length of the normalised Fourier magnitude spectrum of a speed profile, taken
    over the band below a cutoff and above an amplitude threshold. Returned negated by
    convention, so that a value nearer zero is smoother and a large negative value is
    rough. Preferred over a jerk based measure because it is robust to measurement noise
    and independent of duration, whereas jerk measures vary counterintuitively with both.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < 16 or np.allclose(signal, signal[0]):
        return float("nan")
    n = int(2 ** np.ceil(np.log2(signal.size)) * 4)
    spectrum = np.abs(np.fft.rfft(signal, n=n))
    frequency = np.fft.rfftfreq(n, d=dt)
    spectrum = spectrum / max(spectrum.max(), 1e-12)
    band = frequency <= cutoff
    spectrum, frequency = spectrum[band], frequency[band]
    keep = np.flatnonzero(spectrum >= amplitude_threshold)
    if keep.size < 2:
        return float("nan")
    spectrum = spectrum[keep[0]:keep[-1] + 1]
    frequency = frequency[keep[0]:keep[-1] + 1]
    df = (frequency[-1] - frequency[0]) or 1.0
    return float(-np.sum(np.sqrt((np.diff(frequency) / df) ** 2 + np.diff(spectrum) ** 2)))


def smoothness_statistics(data: GaitData) -> list[dict]:
    """Jerk, first differences, and the spectral arc length of the principal signals.

    The record describes the joint traces as impulse trains rather than continuous
    signals, with accelerations reaching 1900 rad/s^2 at the 99th percentile and torque
    changing by 470 N·m within one control step, and neither the action rate penalty nor
    the torque rate penalty measures that character against any standard a reader can
    compare. These do. A caution belongs with them, that a smoothness metric which
    improves under a reward penalty and not under policy space regularisation [15] is
    measuring the penalty rather than the gait.
    """
    group = "smoothness"
    out = []
    torque_rate = np.diff(data.joint_torques, axis=0) / data.dt
    out += summarise(group, "torque rate, all joints", torque_rate, "N·m/s", absolute=True)
    out += summarise(group, "torque first difference, all joints",
                     np.diff(data.joint_torques, axis=0), "N·m", absolute=True)
    jerk = np.diff(data.joint_accelerations, axis=0) / data.dt
    out += summarise(group, "joint jerk, all joints", jerk, "rad/s^3", absolute=True)
    # Spectral arc length of the base speed profile, computed within each episode segment
    # so that the reset discontinuity does not enter the spectrum as a broadband impulse.
    speed = np.linalg.norm(data.base_linear_velocity[..., :2], axis=-1)
    sparcs = []
    for e, start, stop in episode_segments(data):
        segment = speed[start:stop, e]
        value = _spectral_arc_length(segment, data.dt)
        if np.isfinite(value):
            sparcs.append(value)
    out += summarise(group, "spectral arc length of base speed", sparcs, "")
    names = data.joint_names or [f"joint {j}" for j in range(data.num_joints)]
    for j, name in enumerate(names):
        values = []
        for e, start, stop in episode_segments(data):
            value = _spectral_arc_length(data.joint_velocities[start:stop, e, j], data.dt)
            if np.isfinite(value):
                values.append(value)
        if values:
            out.append(record(group, f"spectral arc length {name}", "mean",
                              float(np.mean(values)), ""))
    return out


def variability_statistics(data: GaitData) -> list[dict]:
    """Stride to stride coefficients of variation, and the termination record.

    Variability answers a question the percentiles do not, whether the gait is a limit
    cycle or a sequence of recoveries [16], and the record contains exactly the
    observation it would have flagged, a cadence whose median is exactly one second while
    its tenth percentile lies at 0.720.
    """
    group = "variability"
    contact = contact_state(data)
    out = []
    swings = [[] for _ in range(data.num_feet)]
    for e, f, start, stop in swing_segments(contact):
        swings[f].append((stop - start) * data.dt)
    for f, values in enumerate(swings):
        if len(values) > 2:
            name = data.feet_names[f] if data.feet_names else f"foot {f}"
            out.append(record(group, f"swing duration {name}", "coefficient of variation",
                              100.0 * np.std(values) / max(np.mean(values), 1e-9), "pct"))
    if data.feet_distance is not None:
        lateral = np.abs(data.feet_distance[..., 1])
        out.append(record(group, "step width", "coefficient of variation",
                          100.0 * np.std(lateral) / max(np.mean(lateral), 1e-9), "pct"))
    # Episode outcomes. A policy compared on its behaviour alone is compared on the
    # episodes it survived, which is a selection effect, so the survival record belongs
    # beside every behavioural statistic. The twenty second pass records low height
    # terminations nearly doubling while every impact statistic improved.
    if data.episode_dones is not None:
        out.append(record(group, "episode ends", "count",
                          float(np.sum(data.episode_dones)), ""))
        if data.episode_terminated is not None:
            total = max(float(np.sum(data.episode_dones)), 1.0)
            out.append(record(group, "episode ends", "fraction terminated early",
                              100.0 * float(np.sum(data.episode_terminated)) / total, "pct"))
            out.append(record(group, "episode ends", "fraction timed out",
                              100.0 * float(np.sum(data.episode_time_outs)) / total, "pct"))
        lengths = [(stop - start) * data.dt for _, start, stop in episode_segments(data)]
        out += summarise(group, "episode duration", lengths, "s")
    return out


def reward_budget_statistics(data: GaitData) -> list[dict]:
    """Shares of the positive and negative budgets, moments, and unweighted values.

    THE UNIT TRAP. Every per term series in rewards.npy is a RATE PER SECOND equal to
    func x weight, because RewardManager.compute divides by dt before storing into
    _step_reward at reward_manager.py:157, whereas `total_reward` is the PER STEP value
    equal to the sum of those rates times dt. Verified on run 2026-08-03_11-19-11, the
    mean total is 0.6011 against a summed term mean of 60.115, a ratio of exactly 0.01.
    A budget that adds the two is wrong by a factor of one over the control period.

    The unweighted value, func = rate / weight, is the quantity that makes two runs with
    different weights comparable, and it is the quantity the record inverts by hand in
    every pass from the fifth onward. It requires the weight, and the weight must be the
    one the replay APPLIED rather than one assumed, because play.py has historically
    rebuilt its environment from the live source tree through parse_env_cfg so the tree's
    weight and the trained weight may differ, as they did for run 2026-07-31_10-21-10.
    The weight is therefore logged into rewards.npy under "_weights" at play time, and
    since section 4.4 configures every replay from the run's own params/env.yaml, the
    weight logged IS the weight the run was trained under except where that file was
    absent, in which case the log records the tree's weight and says so by recording what
    was applied. Where no weight is available at all the unweighted records are omitted
    and the weighted ones stand alone.
    """
    group = "rewards"
    if not data.rewards:
        return []
    terms = {k: v for k, v in data.rewards.items() if k != "total_reward"}
    if not terms:
        return []
    means = {k: float(np.mean(v)) for k, v in terms.items()}
    positive = sum(m for m in means.values() if m > 0) or 1.0
    negative = sum(m for m in means.values() if m < 0) or -1.0
    net = positive + negative

    out = [
        record(group, "positive budget", "rate", positive, "1/s"),
        record(group, "negative budget", "rate", negative, "1/s"),
        record(group, "net budget", "rate", net, "1/s"),
        record(group, "term count", "value", len(terms), ""),
    ]
    if "total_reward" in data.rewards:
        out += summarise(group, "total reward per step", data.rewards["total_reward"], "")
        out.append(record(group, "total reward", "rate",
                          float(np.mean(data.rewards["total_reward"])) / data.dt, "1/s"))
    for name, series in sorted(terms.items(), key=lambda kv: -abs(np.mean(kv[1]))):
        mean = means[name]
        out += summarise(group, f"{name}", series, "1/s")
        out.append(record(group, name, "share of total budget",
                          100.0 * mean / (positive - negative), "pct"))
        out.append(record(group, name, "share of its own side",
                          100.0 * mean / (positive if mean > 0 else negative), "pct"))
        weight = data.reward_weights.get(name)
        if weight:
            # The unweighted function value, which is what makes two configurations
            # with different weights comparable at all.
            out += summarise(group, f"{name}, unweighted", series / weight, "")
            out.append(record(group, name, "weight", weight, ""))
        else:
            out.append(record(group, name, "weight unavailable", np.nan, ""))
    return out


_FAMILIES = (
    ("tracking", tracking_statistics),
    ("temporal", temporal_statistics),
    ("impact", contact_event_statistics),
    ("swing", swing_profile_statistics),
    ("feet", foot_statistics),
    ("stance", feet_distance_statistics),
    ("joints", joint_statistics),
    ("energetics", energetic_statistics),
    ("posture", posture_statistics),
    ("stability", stability_statistics),
    ("symmetry", symmetry_statistics),
    ("smoothness", smoothness_statistics),
    ("variability", variability_statistics),
    ("odometry", odometry_statistics),
    ("rewards", reward_budget_statistics),
)


def compute_all(data: GaitData, families: tuple[str, ...] | None = None) -> dict:
    """Every statistic, as a flat record set with metadata.

    A family that raises is caught and reported rather than aborting the whole
    computation, because this runs at the end of a play whose rollout cost minutes and
    whose raw dump must be written regardless. The failure is recorded as a record so
    that a reader of the table learns which family is missing and why.
    """
    records, failures = [], {}
    for name, function in _FAMILIES:
        if families is not None and name not in families:
            continue
        try:
            records.extend(function(data))
        except Exception as error:  # noqa: BLE001
            failures[name] = repr(error)
            records.append(record(name, "family failed", "error", np.nan, ""))
    return {
        "records": records,
        "meta": {
            "num_steps": data.num_steps,
            "num_envs": data.num_envs,
            "num_joints": data.num_joints,
            "num_feet": data.num_feet,
            "dt": data.dt,
            "body_mass": data.body_mass,
            "joint_names": list(data.joint_names),
            "feet_names": list(data.feet_names),
            "has_sole_clearance": data.has_sole_clearance,
            "has_orientation": data.base_quaternion is not None,
            "has_limits": data.joint_effort_limits is not None,
            # The frame the per body vector channels are in. "base_yaw" on the three feet
            # channels means their fore and aft and lateral components are interpretable.
            # An absent entry means the dump predates 2026-08-07 and is world frame, in
            # which case only the norms and the vertical components may be read.
            "frame_convention": dict(data.frame_convention or {}),
            "feet_axes_are_base_frame":
                (data.frame_convention or {}).get("feet_distance") == "base_yaw",
            "has_peak_contact_force": data.has_peak_contact_force,
            "contact_force_history_length": data.contact_force_history_length,
            "contact_force_threshold": CONTACT_FORCE_THRESHOLD,
            "impact_window_s": IMPACT_WINDOW_S,
            "percentiles": list(PERCENTILES),
            "failures": failures,
        },
    }
