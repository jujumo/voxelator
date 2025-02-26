import numpy as np
from stl import mesh
from skimage import measure
import os.path as path
from rich.progress import track
from typing import Optional, Tuple
from jsonargparse import CLI
from voxelator.operators import voxel2mesh


def voxel2stl(
        stl_filepath: str,
        voxel_grid: np.ndarray,
        level_value: float = 1.0,
        scale: float = 1.0
):
    stl_mesh = voxel2mesh(voxel_grid=voxel_grid, level_value=level_value, scale=scale)
    stl_mesh.save(stl_filepath)


def voxel2stl_cli():
    CLI(voxel2stl)


if __name__ == '__main__':
    voxel2stl_cli()
