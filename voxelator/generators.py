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
    assert len(grid_size) == 3
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
    scale: Union[float, Tuple[float, float]] = 1.0,
    shift: Union[float, Tuple[float, float]] = 0.0,
):
    if isinstance(grid_size, int):
        grid_size = grid_size, grid_size, grid_size
    if isinstance(scale, (float, int)):
        scale = scale, scale
    if isinstance(shift, (float, int)):
        shift = shift, shift

    grid_size = np.array(grid_size)
    x_range = np.linspace(-1, 1, grid_size[0]) / scale[0] + shift[0]
    y_range = np.linspace(-1, 1, grid_size[1]) / scale[1] + shift[1]
    x_coord, y_coord = np.meshgrid(x_range, y_range)
    dist_center = np.sqrt(x_coord * x_coord + y_coord * y_coord)
    dist_surface = dist_center - 1.0
    voxel_grid = dist_surface
    voxel_grid = np.expand_dims(voxel_grid, axis=2)
    voxel_grid = np.repeat(voxel_grid, grid_size[2], axis=2)
    return voxel_grid


def generate_voxel_grid_sphere(
    grid_size: Union[int, Tuple[int, int, int]],
    scale: Union[float, Tuple[float, float, float]] = 1.0,
    shift: Union[float, Tuple[float, float, float]] = 0.0,
):
    if isinstance(grid_size, int):
        grid_size = grid_size, grid_size, grid_size
    if isinstance(scale, (float, int)):
        scale = scale, scale, scale
    if isinstance(shift, (float, int)):
        shift = shift, shift, shift
    grid_size = np.array(grid_size)
    x_range = np.linspace(-1, 1, grid_size[0]) / scale[0] + shift[0]
    y_range = np.linspace(-1, 1, grid_size[1]) / scale[1] + shift[2]
    z_range = np.linspace(-1, 1, grid_size[2]) / scale[2] + shift[2]
    x_coord, y_coord, z_coord = np.meshgrid(x_range, y_range, z_range)

    dist_center = np.linalg.norm([x_coord, y_coord, z_coord], axis=0)
    voxel_grid = dist_center - 1.0
    return voxel_grid


def generate_voxel_cubes(
    grid_size: Union[int, Tuple[int, int, int]],
    nb_cubes=80,
    cube_sizes=[3, 5, 7, 7, 9]
):
    if isinstance(grid_size, int):
        grid_size = grid_size, grid_size, grid_size
    radius = grid_size[0] // 2 - 5
    voxel_grid = np.ones(grid_size, dtype=float)
    for i in range(nb_cubes):
        size = np.random.choice(cube_sizes)
        size2 = size // 2
        # x_center = np.random.randint(low=0+size2, high=voxel_size[0]-size2)
        # y_center = np.random.randint(low=0+size2, high=voxel_size[1]-size2)
        z_center = np.random.randint(low=0+size2, high=grid_size[2]-size2)
        theta = np.random.random() * 2 * np.pi

        x_center = int(np.sin(theta) * radius + grid_size[0]/2)
        y_center = int(np.cos(theta) * radius + grid_size[1]/2)
        voxel_grid[x_center-size2: x_center+size2, y_center - size2: y_center + size2, z_center-size2: z_center+size2] = -1.
    return voxel_grid
