"""Frames at matched instants from the three play videos, with the command at each instant.

Every replay is seeded at 42 and configured from its own run's dump, so the command series
is identical across the three and a frame taken at one time is directly comparable. The
instants are chosen from the command record rather than by eye.
"""
import os, subprocess, sys
import numpy as np
import imageio_ffmpeg

EVAL = "/workspace/isaaclab/logs/rsl_rl/kscale_flat_eval"
OUT = ("/workspace/isaaclab/biped/context/artefacts/"
       "kscale_flat-2026-09-24_to_09-25-energy-ankle-ablation/frames")
JOBS = {"A": "2026-09-24_05-18-33_model_9000",
        "B": "2026-09-25_08-00-34_model_8000",
        "C": "2026-09-25_11-29-39_model_8000"}
TIMES = [6.0, 12.0, 20.0, 26.0]
FF = imageio_ffmpeg.get_ffmpeg_exe()

d = np.load(os.path.join(EVAL, JOBS["A"], "data", "42", "dump.npy"),
            allow_pickle=True).item()
cmd = np.asarray(d["commanded_linear_velocity"])[:, 0, :]
cyaw = np.asarray(d["commanded_angular_velocity"])[:, 0]
print("the command of environment zero at each instant")
for t in TIMES:
    i = int(t / 0.01)
    print(f"  t={t:5.1f} s  vx {cmd[i,0]:+.3f}  vy {cmd[i,1]:+.3f}  wz {cyaw[i]:+.3f}")

os.makedirs(OUT, exist_ok=True)
for tag, job in JOBS.items():
    src = os.path.join(EVAL, job, "videos", "play", "42", "rl-video-step-0.mp4")
    if not os.path.exists(src):
        print(f"  MISSING {src}")
        continue
    for t in TIMES:
        dst = os.path.join(OUT, f"{tag}_t{t:04.1f}.png")
        subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(t), "-i", src,
                        "-frames:v", "1", dst], check=True)
        crop = os.path.join(OUT, f"{tag}_t{t:04.1f}_zoom.png")
        subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(t), "-i", src,
                        "-frames:v", "1", "-vf",
                        "crop=iw/2.2:ih/1.6:iw/2.6:ih/2.6,scale=900:-1", crop], check=True)
    print(f"  wrote {len(TIMES)} frames and crops for {tag}")
