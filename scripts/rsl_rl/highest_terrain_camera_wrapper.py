"""Gym wrapper that keeps the viewport camera focused on the environment
with the lowest current terrain difficulty level.

Place this wrapper between the base env and RecordVideo so the camera
is updated before RecordVideo captures the rendered frame.
"""

import gymnasium as gym


class HighestTerrainCameraWrapper(gym.Wrapper):
    """After every step and reset, moves the viewport camera to whichever
    parallel environment currently sits on the lowest terrain difficulty level.

    For flat/plane terrains with no curriculum, the wrapper is a no-op.
    """

    prev_index = -1

    def step(self, action):
        result = self.env.step(action)
        self._track_highest_terrain()
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._track_highest_terrain()
        return result

    def _track_highest_terrain(self):
        base_env = self.unwrapped

        terrain = getattr(getattr(base_env, "scene", None), "terrain", None)
        if terrain is None:
            print("got no terrain so no updating view port")
            return

        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is None:
            return  # flat / plane terrain — no curriculum levels

        highest_idx = int(terrain_levels.argmax().item())

        vcc = getattr(base_env, "viewport_camera_controller", None)
        if vcc is None:
            return

        # Set the target environment index, then track the robot root.
        # update_view_to_asset_root() sets origin_type = "asset_root" internally
        # and the post-update callback will keep following it every render tick.
        vcc.cfg.env_index = highest_idx
        vcc.update_view_to_asset_root("robot")
        if self.prev_index != highest_idx:
            self.prev_index = highest_idx
            print(f"tracking index {self.prev_index}")
