import numpy as np
from typing import Union, Tuple, Optional


def generate_voxel_grid_gyroid(
    grid_size: Union[int, Tuple[int, int, int]],
    grid_periods: Union[float, Tuple[float, float, float]] = 1,
    grid_shifts: Optional[Tuple[float, float, float]] = None
):
    """
    Generate a voxel grid containing gyroid function value in each cell.
    grid_size: width X depth X height number of cells. If a single number is given, apply for all.
    grid_periods:
    grid_shifts:
    """

    if isinstance(grid_size, int):
        grid_size = grid_size, grid_size, grid_size
    assert isinstance(grid_size, tuple)
    grid_size = np.array(grid_size)

    if isinstance(grid_periods, (float, int)):
        scale = grid_periods * np.pi * 2 / grid_size[0]
        scales = np.array([scale, scale, scale])
    else:
        grid_periods = np.array(grid_periods)
        scales = grid_periods * np.pi * 2 / grid_size
    scales = scales.reshape(3, 1, 1, 1)

    indices = np.indices(dimensions=grid_size)
    # center values : -n ... 0 ... n
    indices = indices - grid_size.reshape(3, 1, 1, 1)//2
    # rescale to values
    indices = indices * scales
    if grid_shifts is not None:
        shift = np.array(grid_shifts).reshape(3, 1, 1, 1)
        indices = indices + shift
    xv, yv, zv = indices
    voxel_grid = np.sin(xv) * np.cos(yv) + np.sin(yv) * np.cos(zv) + np.sin(zv) * np.cos(xv)
    return voxel_grid


def generate_voxel_grid_cylinder(
    grid_size: Union[int, Tuple[int, int, int]],
    grid_scale: Union[float, Tuple[float, float, float]] = 1.0,
    grid_shifts: Optional[Tuple[float, float, float]] = None,
):
    if isinstance(grid_size, int):
        grid_size = grid_size, grid_size, grid_size
    grid_size = np.array(grid_size)
    x = np.linspace(-1, 1, grid_size[0])
    y = np.linspace(-1, 1, grid_size[1])
    xv, yv = np.meshgrid(x, y)
    voxel_grid = np.sqrt(xv * xv + yv * yv)
    voxel_grid = np.expand_dims(voxel_grid, axis=2)
    voxel_grid = np.repeat(voxel_grid, grid_size[2], axis=2)
    return voxel_grid
