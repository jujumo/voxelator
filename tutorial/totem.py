import numpy as np
from voxelator.generators import generate_voxel_grid_cylinder, generate_voxel_grid_gyroid
from voxelator.operators import padding, sigmoid
from voxelator.convertors import voxel2surface, surface2open3D, surface2stl
from voxelator.display import display_open3d_window

from jsonargparse import CLI


def create_totem(
    stl: str = 'totem.stl',
    definition: int = 200,
    thickness: float = 0.4
):
    grid_size = np.array([1, 1, 2]) * definition + 1
    gyroid_shift = 0, np.pi/2, 0

    voxel_cylinder = generate_voxel_grid_cylinder(grid_size=grid_size, scale=1.0)
    voxel_gyroid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=2.5, grid_shifts=gyroid_shift)

    voxel_limit = sigmoid(30*voxel_cylinder)
    voxel_grid = (1.0-voxel_limit) * (np.abs(voxel_gyroid)) + voxel_limit - thickness / 2.
    voxel_grid = padding(voxel_grid, padding_size=2, padding_value=2)

    surface = voxel2surface(voxel_grid, level=0.0)
    surface2stl(stl, surface, scale=60./grid_size[0])
    mesh3d = surface2open3D(surface)
    display_open3d_window(mesh3d)


def create_totem_cli():
    CLI(create_totem)


if __name__ == '__main__':
    create_totem_cli()

