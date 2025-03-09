import matplotlib.pyplot as plt
import numpy as np
from voxelator.generators import generate_voxel_grid_cylinder, generate_voxel_grid_gyroid
from voxelator.operators import padding, sigmoid
from voxelator.convertors import voxel2trimesh, trimesh2stl
from voxelator.display import display_trimesh
import trimesh
from jsonargparse import CLI
from typing import Optional


def create_totem(
    stl: Optional[str] = None,
    radius_mm: float = 50.,
    definition: int = 100,
    thickness: float = 0.4
):
    grid_size = np.array([1, 1, 2]) * definition + 1
    gyroid_shift = 0, np.pi/2, 0
    scale = 2 * radius_mm / grid_size[0]

    voxel_grid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=2.5, grid_shifts=gyroid_shift)
    voxel_grid = np.abs(voxel_grid) - (thickness / 2.)
    voxel_grid = padding(voxel_grid, padding_value=1.)
    mesh = voxel2trimesh(voxel_grid, level=0.0, scale=scale)

    cylinder = trimesh.creation.cylinder(radius=radius_mm, height=4*radius_mm, sections=32)
    mesh = mesh.intersection(cylinder)
    if stl is not None:
        trimesh2stl(stl, mesh)
    if stl is None:
        display_trimesh(mesh)


def create_totem_cli():
    CLI(create_totem)


if __name__ == '__main__':
    create_totem_cli()

