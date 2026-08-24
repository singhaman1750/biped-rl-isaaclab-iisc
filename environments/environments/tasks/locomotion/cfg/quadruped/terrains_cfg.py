from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
)

# The sub-terrain proportions are those of BERKELEY_MIMIC_TERRAINS_CFG, so that the
# curriculum presents the same mixture to both robots. Only the amplitudes differ, each
# scaled toward the fraction of standing height the biped configuration presents, the
# quadruped standing at 0.292 m against the biped's 0.75 m.
QUADRUPED_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=5.0,
    num_rows=6,
    num_cols=64,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    color_scheme="height",
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.00, 0.3),
            platform_width=2.0, border_width=0.25,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.00, 0.3),
            platform_width=2.0, border_width=0.25,
        ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.05, step_height_range=(0.00, 0.08), step_width=0.25,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05, step_height_range=(0.00, 0.08), step_width=0.25,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "waves": HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.00, 0.12),
            num_waves=4, border_width=0.25,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.00, 0.05),
            noise_step=0.01, border_width=0.25,
        ),
    },
    curriculum=True,
)

QUADRUPED_ROUGH_TERRAINS_PLAY_CFG = QUADRUPED_ROUGH_TERRAINS_CFG.copy()
QUADRUPED_ROUGH_TERRAINS_PLAY_CFG.num_rows = 5
QUADRUPED_ROUGH_TERRAINS_PLAY_CFG.num_cols = 5
QUADRUPED_ROUGH_TERRAINS_PLAY_CFG.curriculum = False
