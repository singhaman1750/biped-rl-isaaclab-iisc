"""Sweep every quadruped run in the log tree, pairing its dumped configuration against
its TensorBoard scalars, to isolate the variable that separates the runs whose robots
fall through the world from those whose robots stand on it.

Written 2026-09-29 for section 8 of ../../quadruped.md. It reads the event files with a
hand rolled TFRecord and protobuf reader so that it needs neither TensorFlow nor Isaac
Sim, and the dumped env.yaml by plain text search so that it needs no Isaac Lab import.
Run as `python3 sweep.py > run_sweep.txt`.
"""

import collections
import glob
import os
import re
import struct

ROOT = "/ws/IsaacLab/logs/rsl_rl/quadruped_flat"


def records(path):
    """Yield the payload of each TFRecord in a TensorBoard event file."""
    data = open(path, "rb").read()
    i, n = 0, len(data)
    while i + 12 <= n:
        length = struct.unpack_from("<Q", data, i)[0]
        i += 12
        if i + length + 4 > n:
            break
        yield data[i : i + length]
        i += length + 4


def varint(buf, i):
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def fields(buf):
    """Yield (field number, wire type, value) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = varint(buf, i)
            yield number, 0, value
        elif wire == 1:
            yield number, 1, buf[i : i + 8]
            i += 8
        elif wire == 2:
            length, i = varint(buf, i)
            yield number, 2, buf[i : i + length]
            i += length
        elif wire == 5:
            yield number, 5, buf[i : i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")


def scalars(path):
    """Return {tag: [(step, value)]} for every simple scalar in an event file."""
    series = collections.defaultdict(list)
    for record in records(path):
        step, summary = 0, None
        for number, wire, value in fields(record):
            if number == 2 and wire == 0:
                step = value
            if number == 5 and wire == 2:
                summary = value
        if summary is None:
            continue
        for number, wire, value in fields(summary):
            if number != 1:
                continue
            tag = simple = None
            for sub_number, sub_wire, sub_value in fields(value):
                if sub_number == 1 and sub_wire == 2:
                    tag = sub_value.decode()
                if sub_number == 2 and sub_wire == 5:
                    simple = struct.unpack("<f", sub_value)[0]
            if tag is not None and simple is not None:
                series[tag].append((step, simple))
    return series


def first(pattern, text, default="?"):
    match = re.search(pattern, text, re.M)
    return match.group(1) if match else default


header = (
    f"{'run':>22} {'terrain':>10} {'asset':>5} {'hscale':>7} {'envs':>6} {'patch':>7} "
    f"{'selfcol':>8} {'solver':>7} {'iters':>6} {'bc_max':>7} {'baseh':>9} {'linvz':>10} {'airtime':>9}"
)
print(header)
print("-" * len(header))
for run in sorted(os.listdir(ROOT)):
    yaml_path = os.path.join(ROOT, run, "params", "env.yaml")
    if not os.path.isfile(yaml_path):
        continue
    events = glob.glob(os.path.join(ROOT, run, "events.out.tfevents.*"))
    if not events:
        continue
    text = open(yaml_path).read()
    series = scalars(sorted(events)[-1])

    def last(tag):
        return series[tag][-1][1] if series.get(tag) else float("nan")

    base_contact = max((v for _, v in series.get("Episode_Termination/base_contact", [])), default=float("nan"))
    solver = [v for v in re.findall(r"solver_(?:position|velocity)_iteration_count: (\S+)", text) if v != "null"]
    terrain = first(r"terrain_type: (\w+)", text)
    asset = "mjcf" if "MjcfFileCfg" in text else "urdf" if "UrdfFileCfg" in text else "?"
    hscale = first(r"^      horizontal_scale: (\S+)", text)
    envs = first(r"^  num_envs: (\S+)", text)
    patch = first(r"gpu_max_rigid_patch_count: (\S+)", text)
    selfcol = first(r"enabled_self_collisions: (\S+)", text)
    iters = len(series.get("Train/mean_episode_length", []))
    print(
        f"{run:>22} {terrain:>10} {asset:>5} {hscale:>7} {envs:>6} {patch:>7} "
        f"{selfcol:>8} {'/'.join(solver[:2]):>7} {iters:>6} {base_contact:>7.3f} "
        f"{last('Episode_Reward/pen_base_height'):>9.4g} "
        f"{last('Episode_Reward/pen_lin_vel_z'):>10.4g} "
        f"{last('Episode_Reward/feet_air_time'):>9.4g}"
    )
