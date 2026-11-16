"""Count the triangles the quadruped terrain generator lays down at each horizontal scale.

Written 2026-09-29 for section 8 of ../../quadruped.md. It reproduces the arithmetic of
Isaac Lab's own generator without importing it, so that it runs without Isaac Sim. The
column assignment follows terrain_generator.py:243-247, the height field grid follows
height_field/hf_terrains.py:56-57, and the triangulation follows
height_field/utils.py:80, which emits two triangles per cell of the grid.
"""

from collections import Counter

# environments/environments/tasks/locomotion/cfg/quadruped/terrains_cfg.py
NAMES = [
    "flat",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "waves",
    "random_rough",
]
PROPORTIONS = [0.3, 0.1, 0.1, 0.05, 0.05, 0.2, 0.2]
HEIGHT_FIELD = {1, 2, 5, 6}  # the four Hf* sub-terrains, the others being trimesh
SIZE = 15.0
NUM_ROWS, NUM_COLS = 6, 64

total = sum(PROPORTIONS)
cumulative, running = [], 0.0
for proportion in PROPORTIONS:
    running += proportion / total
    cumulative.append(running)

columns = [min(j for j, c in enumerate(cumulative) if index / NUM_COLS + 0.001 < c) for index in range(NUM_COLS)]
counts = Counter(columns)
print("columns per sub-terrain, of", NUM_COLS)
for index in sorted(counts):
    print(f"  {NAMES[index]:24s} {counts[index]:3d}")

hf_columns = sum(counts[index] for index in HEIGHT_FIELD)
hf_tiles = hf_columns * NUM_ROWS
print(f"\nheight field columns {hf_columns}, tiles {hf_tiles} of {NUM_COLS * NUM_ROWS}\n")

for horizontal_scale in (0.05, 0.1):
    pixels = int(SIZE / horizontal_scale)
    per_tile = 2 * (pixels - 1) * (pixels - 1)
    print(
        f"horizontal_scale {horizontal_scale}: grid {pixels} by {pixels}, "
        f"{per_tile:,} triangles per tile, {hf_tiles * per_tile:,} triangles and "
        f"{hf_tiles * pixels * pixels:,} vertices in the concatenated collider"
    )
