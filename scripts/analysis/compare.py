"""Comparative evaluation of trained checkpoints across several experiments.

The script answers one question deterministically. Given a set of training runs, each
holding many periodic checkpoints, which checkpoint walks with the least hip movement,
the least foot splay and the least joint torque saturation, the last weighted towards
the foot joints. It answers it by replaying every qualifying checkpoint through
scripts/rsl_rl/play.py, gathering the dumps into one directory that
scripts/analysis/dashboard.py can read whole, and ranking them on metrics computed from
those dumps.

It runs from the development container and executes the replays inside the two Isaac Lab
containers, scheduling them against the free memory of each container's GPU and against a
ceiling on the number of concurrent processes. The workspace is bind mounted into all
three containers, so a dump a replay writes inside a GPU container is visible here the
moment it lands, and no file is transferred anywhere.

STAGING RATHER THAN COPYING. play.py takes its output directory from the directory of the
checkpoint it is given, at play.py:671, and names the leaf for the seed. Two checkpoints of
one experiment replayed at one seed therefore write to one path, so replaying in the
experiment directory and copying afterwards would have every checkpoint of a run overwrite
its predecessor, and would additionally destroy the evaluation already standing in that
directory. Each job is accordingly given its own staging directory under the comparison
root, holding a link to the checkpoint and a copy of the run's params, and play.py writes
its dump, its statistics and its video directly into the place the comparison wants them.
The resulting layout is root/<experiment>_<checkpoint>/data/<seed>/dump.npy, which is
exactly the layout dashboard.py:228 globs for, with the job name serving as the experiment
label in every panel.

The metrics are computed here rather than read from the statistics record set play.py
writes, for one channel, and read from the dump for the rest. The record set's
"foot yaw misalignment" is formed from an Euler decomposition of the foot quaternion at
stats.py:764, which assumes the foot link's forward axis is its own x and its vertical its
own z. That is the SD_BRS1 convention and not the KScale's, whose foot link carries its
width on x, its vertical on y and its fore and aft length on z, so the record reads 86 to
94 degrees of misalignment where the true figure is 9, an error the reward implementation
already documents at rewards.py:1250 and corrects with its forward_axis argument. This
script therefore reproduces the reward's own convention, taking the toe direction as the
foot link's minus z rotated into the world, and leaves stats.py untouched.

Usage:
    python3 compare.py all
    python3 compare.py plan
    python3 compare.py run
    python3 compare.py analyse
    python3 compare.py report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

# --- Workspace layout -------------------------------------------------------------

# Default experiments under comparison, as directory names beneath LOG_ROOT.
DEFAULT_EXPERIMENTS = (
    "2026-09-24_05-18-33",
    "2026-09-25_08-00-34",
    "2026-09-25_11-29-39",
    "2026-09-28_10-25-57",
    "2026-09-29_06-15-58"
)

LOG_ROOT = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
EVAL_ROOT = "/ws/IsaacLab/logs/rsl_rl/kscale_flat_eval"
REPO_ROOT = "/ws/tron1-rl-isaaclab-cozum"

# The same trees as the GPU containers see them. Both are bind mounts of the same host
# directories, established by docker inspect against the two containers, so a path is
# translated by prefix substitution and nothing is ever copied between containers.
PATH_MAP = (
    ("/ws/IsaacLab/logs", "/workspace/isaaclab/logs"),
    ("/ws/tron1-rl-isaaclab-cozum", "/workspace/isaaclab/biped"),
)

CONTAINERS = ("isaac-lab-base-shreyas-gpu0", "isaac-lab-base-shreyas-gpu1")

# The interpreter inside a GPU container that carries numpy without starting a simulator.
# The analysis phase needs numpy and nothing else, so it is delegated here rather than
# through isaaclab.sh, which would launch Isaac Sim to read a numpy file.
CONTAINER_PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"
CONTAINER_ISAACLAB = "/workspace/isaaclab/isaaclab.sh"
CONTAINER_WORKDIR = "/workspace/isaaclab"
CONTAINER_PLAY = "biped/scripts/rsl_rl/play.py"

# --- Evaluation defaults ----------------------------------------------------------

TASK = "Isaac-Limx-Kscale-Blind-Flat-v0"
SEED = 42
NUM_ENVS = 32
VIDEO_LENGTH = 3000

# Measured per replay of this task at 32 environments with video recording on, by
# nvidia-smi --query-compute-apps inside each container while eight were running. The
# figure is a property of the device and not of the task, an RTX A6000 taking 5504 MiB
# and an RTX 6000 Ada taking 6226 for the identical replay. The larger is rounded up and
# used for both, so that the projection errs towards leaving memory free on whichever
# device the job lands.
JOB_MEMORY_MB = 6300
MEMORY_CAP_MB = 40000
MAX_PROCS_PER_GPU = 5

# A launched replay reaches its full allocation only after Isaac Sim has started, which
# takes upwards of a minute, so its memory is invisible to nvidia-smi for that interval.
# Until a job has been running this long its demand is added to the measured figure by
# hand, without which the scheduler would launch the whole queue into apparently free
# memory and exhaust the device.
SETTLE_SECONDS = 150
POLL_SECONDS = 20
JOB_TIMEOUT_SECONDS = 5400

# Two replays must not sit in the URDF to USD conversion at the same moment, so a launch
# is held until the one before it has left that window. The cause is a name collision
# that a deterministic seed converts from a remote possibility into a reliable event.
# AssetConverterBase names its cache /tmp/IsaacLab/usd_{date}_{time}_{random} at
# asset_converter_base.py:69, and ManagerBasedEnv seeds python's random at
# manager_based_env.py:96 before building the scene at :142, so every replay at one seed
# draws one number and the directory varies only by the second of the spawn. Two replays
# landing in the same second then share a cache, and since the converter writes its hash
# before converting, at asset_converter_base.py:101, the loser reads a matching hash
# beside a 74 MB asset file still being written and spawns a robot with no rigid bodies.
# The symptom is a ValueError from activate_contact_sensors naming the robot prim, which
# reads as an asset fault and is a race. Waiting for the marker below, which play.py
# prints well after the conversion, closes the window at the cost of about half a minute
# of the pipeline per launch.
LAUNCH_GATE_MARKER = "Starting the simulation. This may take a few seconds. Please wait..."
LAUNCH_GATE_SECONDS = 300
LAUNCH_GATE_POLL = 3

# How many times a job that produced no dump is launched again. One retry is enough for
# the residual race, since a retry launched by itself cannot collide with anything.
RETRIES = 1

# --- Metric configuration ---------------------------------------------------------

# Joint name patterns. This robot names its ankle joints foot_pitch_02 and foot_roll_02,
# so the foot pattern below resolves the ankle complex and not the foot bodies.
HIP_JOINT_PATTERN = r"hip"
HIP_OFF_AXIS_PATTERN = r"hip_(roll|yaw)"
FOOT_JOINT_PATTERN = r"(foot|ankle)"

# The foot link axis that points at the toe, as rewards.py:1250 establishes for this
# robot and as the probe of the 2026-09-24 dump confirms at 9.4 degrees of mean
# misalignment against 87.7 for the x axis the Euler path assumes.
FOOT_FORWARD_AXIS = (0.0, 0.0, -1.0)

# Fraction of the available torque above which a sample counts as saturated. The
# available torque is the DC motor envelope at the joint's instantaneous speed where the
# run's parameters supply one, and the flat effort ceiling otherwise. The figure matches
# EFFORT_SATURATION_FRACTION at stats.py:79 so that the two instruments agree.
SATURATION_FRACTION = 0.98

# Relative weight of a foot joint against any other joint in the saturation score, which
# is how "especially in the foot" enters the arithmetic.
FOOT_JOINT_WEIGHT = 2.0

CONTACT_FORCE_THRESHOLD = 1.0

# A checkpoint whose robot falls this often is excluded from the ranking rather than
# allowed to win it, since a machine on the ground moves its hips very little. The unit
# is terminations per environment per second of simulated time, and the threshold is
# calibrated in the report the script writes.
MAX_TERMINATION_RATE = 0.05


def to_container(path: str) -> str:
    """Translate a path in this container to the same file in a GPU container."""
    for here, there in PATH_MAP:
        if path == here or path.startswith(here + os.sep):
            return there + path[len(here):]
    raise ValueError(
        f"{path!r} lies outside every bind mount shared with the GPU containers, "
        f"which are {', '.join(here for here, _ in PATH_MAP)}"
    )


def checkpoint_iteration(name: str) -> int | None:
    """The iteration a checkpoint file name encodes, or None where it encodes none."""
    match = re.fullmatch(r"model_(\d+)\.pt", name)
    return int(match.group(1)) if match else None


class Job:
    """One checkpoint replay, from its discovery to its ranked metrics.

    The staging directory is the single place every stage of the pipeline meets. It is
    created by stage(), filled by the replay, read by the analysis and globbed by the
    dashboard, and its name is the label the comparison carries throughout.
    """

    def __init__(self, experiment: str, iteration: int, checkpoint: str, eval_root: str):
        self.experiment = experiment
        self.iteration = iteration
        self.checkpoint = checkpoint          # absolute path to the trained model
        self.name = f"{experiment}_{os.path.basename(checkpoint)[:-3]}"
        self.directory = os.path.join(eval_root, self.name)
        self.container: str | None = None
        self.launched_at: float | None = None
        self.finished_at: float | None = None
        self.attempts = 0

    # -- the files the pipeline keys on ------------------------------------------

    @property
    def log(self) -> str:
        return os.path.join(self.directory, "play.log")

    @property
    def exit_marker(self) -> str:
        return os.path.join(self.directory, "exit_code")

    @property
    def dump(self) -> str:
        return os.path.join(self.directory, "data", str(SEED), "dump.npy")

    @property
    def statistics(self) -> str:
        return os.path.join(self.directory, "data", str(SEED), "statistics.npy")

    @property
    def video(self) -> str:
        return os.path.join(self.directory, "videos", "play", str(SEED),
                            "rl-video-step-0.mp4")

    @property
    def analysis(self) -> str:
        return os.path.join(self.directory, "analysis.json")

    # -- state -------------------------------------------------------------------

    def is_complete(self) -> bool:
        """True where a replay has already produced the dump this comparison needs.

        Keyed on the artefact rather than on the exit code, so that a rerun of the
        script skips work already done and a job whose marker was lost but whose dump
        survived is not repeated.
        """
        return os.path.exists(self.dump)

    def exit_code(self) -> int | None:
        try:
            with open(self.exit_marker) as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            return None

    def __repr__(self) -> str:
        return f"<Job {self.name}>"


def discover(experiments, log_root: str, eval_root: str, min_iteration: int, max_iteration: int,
             inclusive: bool = True) -> list[Job]:
    """Every checkpoint of every experiment at or beyond the given iteration.

    The order is by experiment and then by iteration, which is the order the schedule,
    the report and the ranking all use, so that two invocations against one tree produce
    one answer.

    Args:
        experiments: Directory names beneath log_root.
        log_root: The tree holding the training runs.
        eval_root: The tree the staging directories are created in.
        min_iteration: The iteration a checkpoint must reach to qualify.
        inclusive: Whether a checkpoint at exactly min_iteration qualifies. True by
            default, so that "after 8000 iterations" admits the 8000 checkpoint.
    """
    jobs = []
    for experiment in experiments:
        directory = os.path.join(log_root, experiment)
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"No experiment directory at {directory}")
        found = []
        for entry in sorted(os.listdir(directory)):
            iteration = checkpoint_iteration(entry)
            if iteration is None:
                continue
            if (iteration > min_iteration and iteration < max_iteration) or (inclusive and (iteration == min_iteration or iteration == max_iteration)):
                found.append((iteration, os.path.join(directory, entry)))
        for iteration, path in sorted(found):
            jobs.append(Job(experiment, iteration, path, eval_root))
    return jobs


def stage(job: Job) -> None:
    """Create the staging directory the replay writes into.

    Two things are placed there and nothing else. The checkpoint, hard linked so that the
    45 MB model is not duplicated and so that the link is a path rather than a reference
    that would resolve differently in each container, and a copy of the run's params
    directory, which is what play.py:680 reads to restore the reward set, the actuator
    gains and the action transform the policy was trained under. Without the params the
    replay would silently take the working tree's configuration instead.
    """
    os.makedirs(job.directory, exist_ok=True)
    target = os.path.join(job.directory, os.path.basename(job.checkpoint))
    if not os.path.exists(target):
        try:
            os.link(job.checkpoint, target)
        except OSError:
            shutil.copy2(job.checkpoint, target)
    source_params = os.path.join(os.path.dirname(job.checkpoint), "params")
    target_params = os.path.join(job.directory, "params")
    if os.path.isdir(source_params) and not os.path.isdir(target_params):
        shutil.copytree(source_params, target_params)
    if not os.path.isdir(target_params):
        print(f"[WARNING] {job.name} has no params directory, the replay will take the "
              f"working tree's reward set, gains and action scale instead of the run's")


# --- The containers and their GPUs -------------------------------------------------


def container_memory_mb(container: str) -> int:
    """Memory in use on the GPU the container sees, in MiB.

    Each container is started with one GPU, so nvidia-smi inside it enumerates exactly
    that device and its index there is always zero. The query must be run inside the
    container because this one has no GPU of its own.
    """
    result = subprocess.run(
        ["docker", "exec", container, "nvidia-smi",
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed in {container}: {result.stderr.strip()}")
    return int(result.stdout.strip().splitlines()[0])


def container_play_processes(container: str) -> int:
    """How many replays are running in the container, counted from its process table.

    Used only as a cross check against the scheduler's own bookkeeping, which remains
    authoritative, since a replay launched by another session would be invisible to the
    bookkeeping and visible here.
    """
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-lc",
         "ps -eo args | grep -c '[p]lay[.]py' || true"],
        capture_output=True, text=True, timeout=120,
    )
    try:
        return int(result.stdout.strip() or 0)
    except ValueError:
        return 0


def launch(job: Job, container: str) -> None:
    """Start one replay inside a container, detached, writing its own exit marker.

    The marker is what the scheduler polls. It is written by the shell after the replay
    returns, so its appearance means the process has ended whatever the outcome, and the
    dump's presence is what distinguishes a success from a failure.
    """
    for stale in (job.exit_marker, job.log):
        if os.path.exists(stale):
            os.remove(stale)
    checkpoint = to_container(os.path.join(job.directory,
                                           os.path.basename(job.checkpoint)))
    command = (
        f"cd {CONTAINER_WORKDIR} && "
        f"{CONTAINER_ISAACLAB} -p {CONTAINER_PLAY} "
        f"--task {TASK} --num_envs {NUM_ENVS} --seed {SEED} "
        f"--video --video_length {VIDEO_LENGTH} --enable_cameras --headless "
        f"--device cuda:0 --checkpoint_path {checkpoint} "
        f"> {to_container(job.log)} 2>&1; "
        f"echo $? > {to_container(job.exit_marker)}"
    )
    subprocess.run(
        ["docker", "exec", "-d", container, "bash", "-lc", command],
        check=True, timeout=120,
    )
    job.container = container
    job.launched_at = time.time()
    job.attempts += 1
    print(f"[LAUNCH] {job.name} on {container}, attempt {job.attempts}")


def await_conversion(job: Job, marker: str = LAUNCH_GATE_MARKER,
                     limit: int = LAUNCH_GATE_SECONDS,
                     poll: int = LAUNCH_GATE_POLL) -> bool:
    """Hold until a launched replay has passed the URDF to USD conversion.

    The marker is printed by play.py once the environment is built and the policy is
    about to be loaded, which is well past the conversion, so its appearance means the
    cache directory is no longer being written. An exit marker appearing first means the
    replay has already ended, which releases the gate just as surely.

    Returns True where the marker was seen, and False on an exit or a timeout, in which
    case the caller proceeds anyway rather than stalling the whole schedule on one job.
    """
    deadline = time.time() + limit
    while time.time() < deadline:
        if os.path.exists(job.exit_marker):
            return False
        try:
            with open(job.log, errors="ignore") as handle:
                if marker in handle.read():
                    return True
        except OSError:
            pass
        time.sleep(poll)
    return False


def kill(job: Job) -> None:
    """Stop an overrunning replay, matched on the checkpoint path in its command line.

    The slot a timed out job holds may not be released while its process still holds the
    memory, since the projection would then count that memory as free and the device
    would be over committed. The checkpoint path is unique to the job, every job having
    its own staging directory, so the match cannot reach another replay.
    """
    if job.container is None:
        return
    target = to_container(os.path.join(job.directory,
                                       os.path.basename(job.checkpoint)))
    subprocess.run(
        ["docker", "exec", job.container, "bash", "-lc",
         f"pkill -f {target!r} || true"],
        capture_output=True, text=True, timeout=120,
    )


def schedule(jobs: list[Job], containers, memory_cap: int, max_procs: int,
             job_memory: int, poll: int, settle: int, timeout: int,
             retries: int = RETRIES) -> None:
    """Run every pending job, holding each GPU inside its memory and process budgets.

    The loop is the whole scheduler. A job is launched onto the first container that has
    both a free process slot and enough projected free memory, the projection adding the
    full demand of every job too young to have allocated yet, since such a job is
    invisible to nvidia-smi and would otherwise be counted as free memory and
    over committed. A finished job releases its slot on the next poll, which is the sense
    in which the next evaluation starts once one finishes.
    """
    pending = [job for job in jobs if not job.is_complete()]
    done = [job for job in jobs if job.is_complete()]
    for job in done:
        print(f"[SKIP] {job.name} already carries a dump")
    running: list[Job] = []

    while pending or running:
        # Retire whatever has finished or overrun.
        for job in list(running):
            if os.path.exists(job.exit_marker):
                job.finished_at = time.time()
                running.remove(job)
                done.append(job)
                code, elapsed = job.exit_code(), job.finished_at - job.launched_at
                ok = job.is_complete()
                print(f"[{'DONE' if ok else 'FAIL'}] {job.name} exit {code} "
                      f"in {elapsed / 60:.1f} min"
                      f"{'' if ok else ', no dump was written, see ' + job.log}")
                if not ok and job.attempts <= retries:
                    done.remove(job)
                    pending.append(job)
                    print(f"[RETRY] {job.name} is queued again, attempt "
                          f"{job.attempts + 1} of {retries + 1}")
            elif time.time() - job.launched_at > timeout:
                kill(job)
                running.remove(job)
                done.append(job)
                print(f"[TIMEOUT] {job.name} exceeded {timeout / 60:.0f} min on "
                      f"{job.container}, it was killed and its slot released")

        # Fill whatever slots have opened.
        for container in containers:
            while pending:
                here = [job for job in running if job.container == container]
                if len(here) >= max_procs:
                    break
                unsettled = sum(1 for job in here
                                if time.time() - job.launched_at < settle)
                measured = container_memory_mb(container)
                projected = measured + job_memory * unsettled
                if projected + job_memory > memory_cap:
                    print(f"[HOLD] {container} at {measured} MiB measured, "
                          f"{projected} MiB projected with {unsettled} unsettled, "
                          f"{job_memory} more would pass the {memory_cap} MiB cap")
                    break
                job = pending.pop(0)
                stage(job)
                launch(job, container)
                running.append(job)
                # The gate below serialises the conversion window and nothing else, the
                # replay it waits on continuing to run throughout, so the loss is about
                # half a minute per launch against an eleven minute replay.
                if not await_conversion(job):
                    print(f"[GATE] {job.name} did not reach the marker, either it ended "
                          f"early or it is slow, the schedule proceeds")

        if pending or running:
            print(f"[WAIT] {len(running)} running, {len(pending)} pending, "
                  f"{len(done)} finished")
            time.sleep(poll)

    failures = [job for job in done if not job.is_complete()]
    print(f"[SCHEDULE] {len(done) - len(failures)} of {len(done)} produced a dump")
    for job in failures:
        print(f"[SCHEDULE] no dump for {job.name}, see {job.log}")


# --- Metrics -----------------------------------------------------------------------
#
# Everything below needs numpy and is therefore run inside a GPU container when this
# script is invoked from a container that has none. The delegation is in analyse_all.


def _numpy():
    import numpy as np
    return np


def _quat_rotate(quat, vector):
    """Rotate a fixed vector by a WXYZ quaternion field of shape (..., 4)."""
    np = _numpy()
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    rotation = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(quat.shape[:-1] + (3, 3))
    return rotation @ np.asarray(vector, dtype=rotation.dtype)


def _yaw(quat):
    """Yaw of a WXYZ quaternion field, the z component of an xyz Euler decomposition."""
    np = _numpy()
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(angle):
    np = _numpy()
    return np.arctan2(np.sin(angle), np.cos(angle))


def _matching(names, pattern):
    """Indices of the names a pattern occurs in, in index order."""
    return [i for i, name in enumerate(names) if re.search(pattern, name)]


def hip_metrics(dump, pattern: str = HIP_JOINT_PATTERN,
                off_axis: str = HIP_OFF_AXIS_PATTERN) -> dict:
    """How much the hips move, in four readings that measure different things.

    A single reading would hide the choice being made, so all four are computed and the
    report ranks under each. `excursion` is the quadrature mean over the hip joints of the
    per joint standard deviation, which is movement about whatever posture the policy
    settled on. `deviation` is the root mean square departure from the configured nominal
    pose, which is movement away from the posture the configuration intends and is the
    quantity the three hip deviation penalties price. `span` is the mean peak to peak
    travel. `speed` is the mean absolute joint velocity, which charges a fast small
    oscillation that the other three read as still.

    The off axis variants restrict the same four to the hip roll and hip yaw, the two
    degrees of freedom this robot's configuration asks to stay near zero, and exclude the
    hip pitch, which must swing for the machine to walk at all.
    """
    np = _numpy()
    names = [str(n) for n in dump["joint_names"]]
    position = np.asarray(dump["joint_positions"])
    velocity = np.asarray(dump["joint_velocities"])
    nominal = dump.get("default_joint_pos")
    out = {"joint_names": names}

    for label, expression in (("hip", pattern), ("hip_off_axis", off_axis)):
        index = _matching(names, expression)
        if not index:
            continue
        columns = position[:, :, index]
        out[f"{label}_excursion"] = float(
            np.sqrt(np.mean(np.square(columns.std(axis=(0, 1)))))
        )
        out[f"{label}_span"] = float(
            np.mean(columns.max(axis=(0, 1)) - columns.min(axis=(0, 1)))
        )
        out[f"{label}_speed"] = float(np.mean(np.abs(velocity[:, :, index])))
        if nominal is not None:
            reference = np.asarray(nominal)[index]
            out[f"{label}_deviation"] = float(
                np.sqrt(np.mean(np.square(columns - reference)))
            )
    out["hip_per_joint"] = {
        names[j]: {
            "std": float(position[:, :, j].std()),
            "span": float(position[:, :, j].max() - position[:, :, j].min()),
            "speed": float(np.abs(velocity[:, :, j]).mean()),
        }
        for j in _matching(names, pattern)
    }
    return out


def splay_metrics(dump, forward_axis=FOOT_FORWARD_AXIS,
                  threshold: float = CONTACT_FORCE_THRESHOLD) -> dict:
    """Foot splay, taken as the toe direction against the base heading.

    The toe direction is the foot link's forward axis rotated into the world and projected
    onto the horizontal, which is the construction feet_yaw_alignment uses at
    rewards.py:1313 once its forward_axis argument is supplied. Its heading is compared
    against the base yaw and the difference wrapped, giving each foot a signed angle.

    Toe-in is that angle with the left foot's sign flipped, so that a positive figure on
    either foot means the toe has turned towards the midline and a pigeon toed machine
    scores positive on both. The dump records its feet left before right, which the
    articulation's own body order gives and which the lateral separation confirms
    independently, that separation being taken as foot zero minus foot one in the base
    frame and standing positive.

    Three summaries are returned because they fail differently. `abs_mean` is the mean
    magnitude over both feet and every sample and is the headline, `signed_mean` says
    which way the splay goes and separates a pigeon toe from a duck foot, and
    `differential` is the angle between the two feet, which is blind to the base heading
    and so survives a turn that the other two charge for.
    """
    np = _numpy()
    feet_quat = np.asarray(dump["feet_quaternions"])
    base_quat = np.asarray(dump["base_quaternion"])
    base_yaw = _yaw(base_quat)
    relative = []
    for foot in range(feet_quat.shape[2]):
        toe = _quat_rotate(feet_quat[:, :, foot, :], forward_axis)
        relative.append(_wrap(np.arctan2(toe[..., 1], toe[..., 0]) - base_yaw))
    # Left foot first. A left toe turned towards the midline yaws negative, a right toe
    # turned towards the midline yaws positive, so the left sign is flipped.
    toe_in = np.degrees(np.stack([-relative[0], relative[1]], axis=-1))
    contact = (np.linalg.norm(np.asarray(dump["feet_contact_forces"]), axis=-1)
               > threshold)
    feet_names = [str(n) for n in dump.get("feet_names", [])]
    out = {
        "toe_in_abs_mean": float(np.abs(toe_in).mean()),
        "toe_in_signed_mean": float(toe_in.mean()),
        "toe_in_p95": float(np.percentile(np.abs(toe_in), 95)),
        "toe_in_differential": float(
            np.abs(np.degrees(_wrap(relative[1] - relative[0]))).mean()
        ),
        "feet_names": feet_names,
        "per_foot": {},
    }
    for foot in range(toe_in.shape[-1]):
        label = feet_names[foot] if foot < len(feet_names) else f"foot {foot}"
        planted = contact[:, :, foot]
        out["per_foot"][label] = {
            "toe_in_mean": float(toe_in[..., foot].mean()),
            "toe_in_abs_mean": float(np.abs(toe_in[..., foot]).mean()),
            "stance_toe_in_mean": float(toe_in[..., foot][planted].mean())
            if planted.any() else float("nan"),
        }
    stance = toe_in[contact] if contact.any() else toe_in
    out["stance_toe_in_abs_mean"] = float(np.abs(stance).mean())
    out["stance_toe_in_signed_mean"] = float(stance.mean())
    return out


def _actuator_envelope(dump, run_directory: str):
    """Per joint DC motor parameters read from the run's own dumped configuration.

    Returns a triple of arrays over the joints, the saturation effort, the effort limit
    and the velocity limit, or None where the run dumped no actuator block. The
    saturation effort is the one quantity of the three that the dump itself does not
    carry, which is the gap gait_metrics.md section 17 records, and the run's params
    supply it.
    """
    np = _numpy()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import experiment_params

    params = experiment_params.load_params(run_directory)
    groups = experiment_params.actuator_groups(params) if params else {}
    if not groups:
        return None
    names = [str(n) for n in dump["joint_names"]]
    saturation = np.full(len(names), np.nan)
    effort = np.full(len(names), np.nan)
    speed = np.full(len(names), np.nan)
    for group in groups.values():
        expressions = group.get("joint_names_expr") or []
        if isinstance(expressions, str):
            expressions = [expressions]
        for j, name in enumerate(names):
            if not any(re.fullmatch(e, name) for e in expressions):
                continue
            if group.get("saturation_effort") is not None:
                saturation[j] = float(group["saturation_effort"])
            if group.get("effort_limit") is not None:
                effort[j] = float(group["effort_limit"])
            if group.get("velocity_limit") is not None:
                speed[j] = float(group["velocity_limit"])
    if not np.isfinite(saturation).any():
        return None
    return saturation, effort, speed


def saturation_metrics(dump, run_directory: str, pattern: str = FOOT_JOINT_PATTERN,
                       fraction: float = SATURATION_FRACTION,
                       foot_weight: float = FOOT_JOINT_WEIGHT) -> dict:
    """How often each joint's torque stands against the torque available to it.

    Two readings are computed. The envelope reading compares the applied torque against
    the DC motor curve at the joint's instantaneous speed, reproducing
    DCMotor._clip_effort at actuator_pd.py:295, and is the correct one, since the
    available torque falls with speed and a flat ceiling therefore understates saturation
    wherever the joint is moving. The ceiling reading compares against the flat effort
    limit and is what stats.py:661 reports, retained so that the two instruments can be
    read against one another.

    Neither reading is exact, because IdentifiedActuator subtracts its friction term after
    the clip at actuator_pd.py:34, so the dumped torque sits a little below whatever
    bound was applied to it. This is why the threshold is a fraction of the bound rather
    than the bound itself, and it is the reason a saturation fraction swings with that
    threshold, which gait_metrics.md records at twenty five points between 98 and 99
    per cent on this robot.

    The score is the weighted mean of the per joint envelope fractions, with the foot
    joints carrying foot_weight against every other joint's one.
    """
    np = _numpy()
    names = [str(n) for n in dump["joint_names"]]
    torque = np.asarray(dump["joint_torques"])
    velocity = np.asarray(dump["joint_velocities"])
    ceiling = dump.get("joint_effort_limits")
    parameters = _actuator_envelope(dump, run_directory)

    per_joint = {}
    for j, name in enumerate(names):
        entry = {}
        column = np.abs(torque[:, :, j])
        if ceiling is not None and np.isfinite(ceiling[j]) and ceiling[j] > 0:
            entry["ceiling"] = float(
                100.0 * np.mean(column >= fraction * float(ceiling[j]))
            )
            entry["p99_of_ceiling"] = float(
                100.0 * np.percentile(column, 99) / float(ceiling[j])
            )
        if parameters is not None:
            saturation, effort, speed = parameters
            if np.isfinite(saturation[j]) and np.isfinite(effort[j]) \
                    and np.isfinite(speed[j]) and speed[j] > 0:
                reach = speed[j] * (1.0 + effort[j] / saturation[j])
                clipped = np.clip(velocity[:, :, j], -reach, reach)
                top = np.minimum(saturation[j] * (1.0 - clipped / speed[j]), effort[j])
                bottom = np.maximum(saturation[j] * (-1.0 - clipped / speed[j]),
                                    -effort[j])
                available = np.where(torque[:, :, j] >= 0.0, top, -bottom)
                entry["envelope"] = float(100.0 * np.mean(
                    column >= np.maximum(fraction * available, 1e-9)
                ))
        per_joint[name] = entry

    foot = _matching(names, pattern)
    out = {"per_joint": per_joint,
           "foot_joints": [names[j] for j in foot],
           "reference": "envelope" if parameters is not None else "ceiling"}
    key = out["reference"]
    values = [(j, per_joint[name].get(key)) for j, name in enumerate(names)]
    values = [(j, v) for j, v in values if v is not None]
    if values:
        weights = [foot_weight if j in foot else 1.0 for j, _ in values]
        out["score"] = float(sum(w * v for w, (_, v) in zip(weights, values))
                             / sum(weights))
        out["all_joint_mean"] = float(sum(v for _, v in values) / len(values))
        foot_values = [v for j, v in values if j in foot]
        if foot_values:
            out["foot_joint_mean"] = float(sum(foot_values) / len(foot_values))
            out["foot_joint_max"] = float(max(foot_values))
    return out


def viability_metrics(dump) -> dict:
    """Whether the replay walked at all, which every other metric presupposes.

    A machine that has fallen holds its hips still, points its toes nowhere in particular
    and asks nothing of its actuators, so it would sweep all three objectives. The
    termination rate and the tracking errors are therefore computed alongside, and the
    ranking excludes a checkpoint that fails the termination gate rather than letting it
    win.
    """
    np = _numpy()
    dt = float(dump.get("step_dt") or 0.01)
    out = {"dt": dt}
    terminated = dump.get("episode_terminated")
    if terminated is not None and len(terminated):
        terminated = np.asarray(terminated)
        steps, envs = terminated.shape[0], terminated.shape[1]
        out["termination_rate"] = float(terminated.sum() / (envs * steps * dt))
        out["terminations"] = int(terminated.sum())
    timeouts = dump.get("episode_time_outs")
    if timeouts is not None and len(timeouts):
        out["timeouts"] = int(np.asarray(timeouts).sum())
    linear = np.asarray(dump["base_linear_velocity"])
    commanded = np.asarray(dump["commanded_linear_velocity"])
    out["planar_tracking_error"] = float(
        np.mean(np.linalg.norm(linear[:, :, :2] - commanded[:, :, :2], axis=-1))
    )
    angular = np.asarray(dump["base_angular_velocity"])
    commanded_yaw = np.asarray(dump["commanded_angular_velocity"])
    if commanded_yaw.ndim == 3:
        commanded_yaw = commanded_yaw[..., 0]
    out["yaw_tracking_error"] = float(np.mean(np.abs(angular[:, :, 2] - commanded_yaw)))
    height = np.asarray(dump["base_com_position"])[:, :, 2]
    out["base_height_mean"] = float(height.mean())
    out["base_height_std"] = float(height.std())
    return out


def analyse_one(job_directory: str, seed: int = SEED) -> dict:
    """Every metric for one staged job, from its dump and its copied params."""
    np = _numpy()
    dump_path = os.path.join(job_directory, "data", str(seed), "dump.npy")
    dump = np.load(dump_path, allow_pickle=True).item()
    return {
        "name": os.path.basename(job_directory.rstrip(os.sep)),
        "dump": dump_path,
        "num_steps": int(np.asarray(dump["joint_positions"]).shape[0]),
        "num_envs": int(np.asarray(dump["joint_positions"]).shape[1]),
        "hip": hip_metrics(dump),
        "splay": splay_metrics(dump),
        "saturation": saturation_metrics(dump, job_directory),
        "viability": viability_metrics(dump),
    }


def staged_jobs(eval_root: str, seed: int = SEED) -> list[str]:
    """Every staged directory under the comparison root that carries a dump."""
    if not os.path.isdir(eval_root):
        return []
    found = []
    for entry in sorted(os.listdir(eval_root)):
        directory = os.path.join(eval_root, entry)
        if os.path.exists(os.path.join(directory, "data", str(seed), "dump.npy")):
            found.append(directory)
    return found


def analyse_all(eval_root: str, container: str | None = None,
                delegate: bool = True, force: bool = False) -> dict:
    """Analyse every staged job, delegating to a GPU container where numpy is absent.

    The development container carries no numpy and its interpreter is externally managed,
    so rather than install into it the analysis is run through a GPU container's own
    interpreter, against the very same files, both trees being bind mounts of one host
    directory. The delegated call is this script under a path translated for that
    container, so there is one implementation of every metric and not two.
    """
    directories = staged_jobs(eval_root)
    if not directories:
        raise FileNotFoundError(f"No staged job under {eval_root} carries a dump")

    try:
        import numpy  # noqa: F401
        local = True
    except ImportError:
        local = False

    if not local:
        if not delegate:
            raise ImportError("numpy is unavailable and delegation was refused")
        container = container or CONTAINERS[-1]
        print(f"[ANALYSE] numpy is absent here, delegating to {container}")
        command = [
            "docker", "exec", container, CONTAINER_PYTHON,
            to_container(os.path.abspath(__file__)), "analyse",
            "--eval-root", to_container(eval_root), "--no-delegate", "--no-report",
        ]
        if force:
            command.append("--force")
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"delegated analysis failed in {container}")
        return load_analyses(eval_root)

    for directory in directories:
        target = os.path.join(directory, "analysis.json")
        if os.path.exists(target) and not force:
            print(f"[ANALYSE] {os.path.basename(directory)} already analysed")
            continue
        print(f"[ANALYSE] {os.path.basename(directory)}")
        record = analyse_one(directory)
        with open(target, "w") as handle:
            json.dump(record, handle, indent=2)
    return load_analyses(eval_root)


def load_analyses(eval_root: str) -> dict:
    """The per job analysis records, keyed by job name, in directory order."""
    records = {}
    for directory in staged_jobs(eval_root):
        path = os.path.join(directory, "analysis.json")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            records[os.path.basename(directory)] = json.load(handle)
    return records


# --- Ranking and reporting ---------------------------------------------------------

# The three objectives, each read from its record and each smaller for better.
OBJECTIVES = (
    ("hip", ("hip", "hip_excursion"), "hip movement, rad"),
    ("splay", ("splay", "toe_in_abs_mean"), "foot splay, deg"),
    ("saturation", ("saturation", "score"), "torque saturation, pct"),
)

# The alternative readings of hip movement, reported beside the ranking so that a reader
# sees at once whether the answer turns on the choice among them.
HIP_ALTERNATIVES = ("hip_excursion", "hip_deviation", "hip_span", "hip_speed",
                    "hip_off_axis_excursion", "hip_off_axis_deviation")
SPLAY_ALTERNATIVES = ("toe_in_abs_mean", "toe_in_p95", "toe_in_differential",
                      "stance_toe_in_abs_mean")
SATURATION_ALTERNATIVES = ("score", "foot_joint_mean", "foot_joint_max",
                           "all_joint_mean")


def _reach(record: dict, path) -> float | None:
    for key in path:
        if not isinstance(record, dict) or key not in record:
            return None
        record = record[key]
    return record if isinstance(record, (int, float)) else None


def _normalise(values: dict) -> dict:
    """Min to max normalisation over the candidate set, smaller staying better.

    Normalising over the candidates rather than against an absolute scale is what makes
    the composite readable, every objective spanning zero to one across the very set
    being compared, and it is deterministic given that set.
    """
    present = [v for v in values.values() if v is not None]
    if not present:
        return {k: None for k in values}
    low, high = min(present), max(present)
    if high - low < 1e-12:
        return {k: (0.0 if v is not None else None) for k, v in values.items()}
    return {k: ((v - low) / (high - low) if v is not None else None)
            for k, v in values.items()}


def rank(records: dict, weights: dict | None = None,
         max_termination_rate: float = MAX_TERMINATION_RATE) -> dict:
    """Score every checkpoint against the three objectives and order them.

    A checkpoint whose termination rate exceeds the gate is carried in the table and
    excluded from the ordering, because the three objectives are all minimised by a
    machine that has stopped walking and a gate is the only thing that prevents a fall
    from reading as a result.
    """
    weights = weights or {name: 1.0 for name, _, _ in OBJECTIVES}
    viable, excluded = {}, {}
    for name, record in records.items():
        rate = _reach(record, ("viability", "termination_rate"))
        if rate is not None and rate > max_termination_rate:
            excluded[name] = rate
        else:
            viable[name] = record

    columns = {}
    for objective, path, _ in OBJECTIVES:
        columns[objective] = {n: _reach(r, path) for n, r in viable.items()}
    normalised = {o: _normalise(v) for o, v in columns.items()}

    scored = []
    for name in viable:
        parts = {o: normalised[o][name] for o, _, _ in OBJECTIVES}
        if any(v is None for v in parts.values()):
            continue
        composite = sum(weights[o] * parts[o] for o, _, _ in OBJECTIVES)
        scored.append({
            "name": name,
            "composite": composite / sum(weights.values()),
            "normalised": parts,
            "raw": {o: columns[o][name] for o, _, _ in OBJECTIVES},
            "termination_rate": _reach(viable[name], ("viability", "termination_rate")),
            "planar_tracking_error": _reach(
                viable[name], ("viability", "planar_tracking_error")),
        })
    scored.sort(key=lambda row: (row["composite"], row["name"]))

    # The position each checkpoint takes under each objective read alone, carried beside
    # the composite so that a reader sees whether the winner wins on all three or is
    # carried by one, which a single normalised number cannot say.
    for objective, _, _ in OBJECTIVES:
        order = sorted((r for r in scored if r["raw"][objective] is not None),
                       key=lambda r: r["raw"][objective])
        for position, row in enumerate(order, start=1):
            row.setdefault("ranks", {})[objective] = position

    # The Pareto front, being every candidate no other candidate beats on all three.
    front = []
    for row in scored:
        dominated = any(
            all(other["raw"][o] <= row["raw"][o] for o, _, _ in OBJECTIVES)
            and any(other["raw"][o] < row["raw"][o] for o, _, _ in OBJECTIVES)
            for other in scored if other is not row
        )
        if not dominated:
            front.append(row["name"])
    return {"ranked": scored, "pareto": front, "excluded": excluded,
            "weights": weights, "gate": max_termination_rate}


def _cell(value, digits: int = 6) -> str:
    """A number for the comparison table, or an empty field where the dump lacked it."""
    return "" if value is None else f"{value:.{digits}f}"


def _table(rows, headers) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(out)


def _alternatives(records: dict, group: str, keys) -> str:
    """The ordering under each alternative reading, best first."""
    lines = []
    for key in keys:
        values = {n: _reach(r, (group, key)) for n, r in records.items()}
        values = {n: v for n, v in values.items() if v is not None}
        if not values:
            continue
        order = sorted(values, key=lambda n: values[n])
        lines.append(f"  {key:28s} {order[0]}  ({values[order[0]]:.4g}), "
                     f"then {', '.join(order[1:4])}")
    return "\n".join(lines)


def report(eval_root: str, records: dict, ranking: dict) -> str:
    """Write the comparison table to disk and return it for printing."""
    headers = ["rank", "checkpoint", "composite", "hip rad", "(r)", "splay deg",
               "(r)", "saturation pct", "(r)", "term/s", "track m/s"]
    rows = []
    for position, row in enumerate(ranking["ranked"], start=1):
        ranks = row.get("ranks", {})
        rows.append([
            position, row["name"], f"{row['composite']:.4f}",
            f"{row['raw']['hip']:.4f}", ranks.get("hip", ""),
            f"{row['raw']['splay']:.3f}", ranks.get("splay", ""),
            f"{row['raw']['saturation']:.2f}", ranks.get("saturation", ""),
            _cell(row["termination_rate"], 4), _cell(row["planar_tracking_error"], 3),
        ])
    text = [_table(rows, headers)]

    # The best checkpoint of each experiment, which is the comparison the brief asks for
    # once the checkpoint within a run has been chosen.
    best_of = {}
    for row in ranking["ranked"]:
        experiment = row["name"].rsplit("_model_", 1)[0]
        if experiment not in best_of:
            best_of[experiment] = row
    text.append("\nBest checkpoint of each experiment:")
    for experiment in sorted(best_of):
        row = best_of[experiment]
        text.append(f"  {experiment}  {row['name'].rsplit('_', 1)[-1]}  "
                    f"composite {row['composite']:.4f}, hip {row['raw']['hip']:.4f} rad, "
                    f"splay {row['raw']['splay']:.3f} deg, "
                    f"saturation {row['raw']['saturation']:.2f} pct")
    if ranking["excluded"]:
        text.append("\nExcluded by the termination gate of "
                    f"{ranking['gate']} per environment second:")
        for name, rate in sorted(ranking["excluded"].items()):
            text.append(f"  {name}  {rate:.4f}")
    text.append("\nPareto front, no other checkpoint beating these on all three:")
    text.append("  " + ", ".join(ranking["pareto"]))
    text.append("\nBest under each alternative reading of hip movement:")
    text.append(_alternatives(records, "hip", HIP_ALTERNATIVES))
    text.append("\nBest under each alternative reading of foot splay:")
    text.append(_alternatives(records, "splay", SPLAY_ALTERNATIVES))
    text.append("\nBest under each alternative reading of torque saturation:")
    text.append(_alternatives(records, "saturation", SATURATION_ALTERNATIVES))
    rendered = "\n".join(text)

    with open(os.path.join(eval_root, "comparison.txt"), "w") as handle:
        handle.write(rendered + "\n")
    with open(os.path.join(eval_root, "comparison.json"), "w") as handle:
        json.dump({"ranking": ranking, "records": records}, handle, indent=2)
    with open(os.path.join(eval_root, "comparison.csv"), "w") as handle:
        handle.write(",".join(["checkpoint", "composite", "hip_excursion",
                               "toe_in_abs_mean", "saturation_score",
                               "termination_rate", "planar_tracking_error"]) + "\n")
        for row in ranking["ranked"]:
            handle.write(",".join([
                row["name"], f"{row['composite']:.6f}",
                f"{row['raw']['hip']:.6f}", f"{row['raw']['splay']:.6f}",
                f"{row['raw']['saturation']:.6f}",
                _cell(row["termination_rate"]), _cell(row["planar_tracking_error"]),
            ]) + "\n")
    return rendered


# --- Entry point -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay every checkpoint of several experiments and rank them on "
                    "hip movement, foot splay and torque saturation.")
    parser.add_argument("phase", choices=("plan", "run", "analyse", "report", "all"),
                        help="plan enumerates and stages, run schedules the replays, "
                             "analyse computes the metrics, report ranks them, all "
                             "performs the four in order.")
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--log-root", default=LOG_ROOT)
    parser.add_argument("--eval-root", default=EVAL_ROOT)
    parser.add_argument("--min-iteration", type=int, default=8000)
    parser.add_argument("--max-iteration", type=int, default=11000)
    parser.add_argument("--exclusive", action="store_true",
                        help="Require a checkpoint to exceed --min-iteration rather "
                             "than reach it, excluding the 8000 checkpoint itself.")
    parser.add_argument("--containers", nargs="+", default=list(CONTAINERS))
    parser.add_argument("--memory-cap", type=int, default=MEMORY_CAP_MB)
    parser.add_argument("--max-procs", type=int, default=MAX_PROCS_PER_GPU)
    parser.add_argument("--job-memory", type=int, default=JOB_MEMORY_MB)
    parser.add_argument("--poll", type=int, default=POLL_SECONDS)
    parser.add_argument("--settle", type=int, default=SETTLE_SECONDS)
    parser.add_argument("--job-timeout", type=int, default=JOB_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=RETRIES,
                        help="How many times a replay that produced no dump is launched "
                             "again before it is given up on.")
    parser.add_argument("--max-termination-rate", type=float,
                        default=MAX_TERMINATION_RATE)
    parser.add_argument("--weights", nargs=3, type=float, metavar=("HIP", "SPLAY", "SAT"),
                        default=[1.0, 1.0, 1.0])
    parser.add_argument("--force", action="store_true",
                        help="Recompute the analysis of a job that already carries one.")
    parser.add_argument("--no-delegate", dest="delegate", action="store_false",
                        help="Refuse to run the analysis inside a GPU container, which "
                             "is what the delegated invocation itself passes.")
    parser.add_argument("--no-report", dest="want_report", action="store_false",
                        help="Compute the metrics without ranking them, which the "
                             "delegated invocation passes so that the report is "
                             "printed once, by the caller, and not twice.")
    args = parser.parse_args()

    os.makedirs(args.eval_root, exist_ok=True)

    if args.phase in ("plan", "run", "all"):
        jobs = discover(args.experiments, args.log_root, args.eval_root,
                        args.min_iteration, args.max_iteration, inclusive=not args.exclusive)
        print(f"[PLAN] {len(jobs)} checkpoints at or beyond iteration "
              f"{args.min_iteration} across {len(args.experiments)} experiments")
        for job in jobs:
            print(f"  {job.name:48s} {'dump present' if job.is_complete() else 'pending'}")
        for job in jobs:
            stage(job)
        if args.phase == "plan":
            for container in args.containers:
                used = container_memory_mb(container)
                print(f"[GPU] {container} {used} MiB in use, "
                      f"{max(0, args.memory_cap - used) // args.job_memory} slots "
                      f"below the {args.memory_cap} MiB cap, "
                      f"{args.max_procs} allowed by the process ceiling")
            return 0
        schedule(jobs, args.containers, args.memory_cap, args.max_procs,
                 args.job_memory, args.poll, args.settle, args.job_timeout,
                 args.retries)

    if args.phase in ("analyse", "all"):
        analyse_all(args.eval_root, args.containers[-1], args.delegate, args.force)

    if args.phase in ("report", "analyse", "all") and args.want_report:
        records = load_analyses(args.eval_root)
        if not records:
            print("[REPORT] no analysis record under the comparison root")
            return 1
        weights = dict(zip([o for o, _, _ in OBJECTIVES], args.weights))
        ranking = rank(records, weights, args.max_termination_rate)
        print()
        print(report(args.eval_root, records, ranking))
        if ranking["ranked"]:
            best = ranking["ranked"][0]
            print(f"\n[BEST] {best['name']} at a composite of {best['composite']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
