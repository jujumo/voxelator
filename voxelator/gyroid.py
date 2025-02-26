import numpy as np
import os.path as path
from jsonargparse import CLI
from typing import Union, Tuple, Optional
from voxelator.voxel2stl import voxel2stl
from voxelator.operators import padding
from voxelator.generators import generate_voxel_grid_gyroid


def gyroid2file(
    filepath: str,
    level_value=0.1,
    periods: float = 3.0,
    width: int = 201,
    depth: Optional[int] = None,
    height: Optional[int] = None,
    surface_mode: bool = True
):
    height = 301
    depth = depth if depth is not None else width
    height = height if height is not None else width
    grid_size = width, depth, height
    shift = np.pi/2, 0, 0
    voxel_grid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=periods, grid_shifts=shift)
    if surface_mode:
        voxel_grid = np.abs(voxel_grid)
    voxel_grid = padding(voxel_grid, padding_size=1, padding_value=2)
    if path.splitext(filepath)[1].upper() == '.STL':
        voxel2stl(stl_filepath=filepath, voxel_grid=voxel_grid, level_value=level_value, scale=0.5)
    else:
        np.save(filepath, voxel_grid)


def create_gyroid_cli():
    CLI(gyroid2file)


if __name__ == '__main__':
    create_gyroid_cli()


