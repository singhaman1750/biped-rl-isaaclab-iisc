"""Read the reward budget of one run out of its TensorBoard event file.

Reuses the hand rolled TFRecord and protobuf reader of sweep.py so that it needs
neither TensorFlow nor Isaac Sim. Prints, for the last recorded iteration, the
mean episode return and every Episode_Reward term, so that a reward weight may be
priced against the budget it competes in.
"""

import collections
import glob
import os
import struct
import sys


def records(path):
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
            return


def scalars(run_dir):
    """tag -> list of (step, value), in file order."""
    out = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))):
        for payload in records(path):
            step = 0
            summary = None
            for number, wire, value in fields(payload):
                if number == 2 and wire == 0:
                    step = value
                elif number == 5 and wire == 2:
                    summary = value
            if summary is None:
                continue
            for number, wire, value in fields(summary):
                if number != 1 or wire != 2:
                    continue
                tag = None
                simple = None
                for n2, w2, v2 in fields(value):
                    if n2 == 1 and w2 == 2:
                        tag = v2.decode("utf8", "replace")
                    elif n2 == 2 and w2 == 5:
                        simple = struct.unpack("<f", v2)[0]
                if tag is not None and simple is not None:
                    out[tag].append((step, simple))
    return out


def report(run_dir, window=50):
    s = scalars(run_dir)
    if not s:
        print(f"{run_dir}: no scalars")
        return
    s = {k: v for k, v in s.items() if not k.endswith("/time")}
    last = max(step for series in s.values() for step, _ in series)
    print(f"=== {os.path.basename(run_dir)}  last step {last} ===")

    def tail_mean(tag):
        series = [v for step, v in s.get(tag, []) if step > last - window]
        return sum(series) / len(series) if series else None

    for tag in ("Train/mean_reward", "Train/mean_episode_length"):
        v = tail_mean(tag)
        if v is not None:
            print(f"  {tag:34s} {v:12.4f}")
    terms = sorted(t for t in s if t.startswith("Episode_Reward/"))
    total_pos = total_neg = 0.0
    rows = []
    for tag in terms:
        v = tail_mean(tag)
        if v is None:
            continue
        rows.append((v, tag))
        if v >= 0:
            total_pos += v
        else:
            total_neg += v
    rows.sort(key=lambda r: -abs(r[0]))
    print(f"  {'term':40s} {'per second':>12s} {'per episode':>12s} {'share of |neg|':>15s}")
    for v, tag in rows:
        share = (abs(v) / abs(total_neg) * 100.0) if (v < 0 and total_neg) else float("nan")
        print(f"  {tag[15:]:40s} {v:12.5f} {v * 20.0:12.4f} {share:15.2f}")
    print(f"  {'SUM positive':40s} {total_pos:12.5f} {total_pos * 20.0:12.4f}")
    print(f"  {'SUM negative':40s} {total_neg:12.5f} {total_neg * 20.0:12.4f}")
    print()


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
