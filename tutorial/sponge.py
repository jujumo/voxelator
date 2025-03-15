import numpy as np
from voxelator.generators import generate_voxel_grid_cylinder, generate_voxel_grid_gyroid
from voxelator.operators import padding, sigmoid
from voxelator.convertors import voxel2trimesh, trimesh2stl
from voxelator.display import display_trimesh
import trimesh
from jsonargparse import CLI
from typing import Optional


def create_sponge(
    stl: Optional[str] = None,
    periods: float = 3.0,
    radius: float = 50.,
    definition: int = 100,
    thickness: float = 0.3,
    verbose: bool = False
):
    grid_size = np.array([1, 1, 1]) * definition + 1
    gyroid_shift = 0, np.pi/2, 0
    scale = 2 * radius / grid_size[0]

    voxel_grid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=periods, grid_shifts=gyroid_shift)
    voxel_grid = np.abs(voxel_grid) - (thickness / 2.)
    voxel_grid = padding(voxel_grid, padding_value=1.)
    mesh = voxel2trimesh(voxel_grid, level=0.0, scale=scale)

    sphere = trimesh.creation.icosphere(subdivisions=1)  # trimesh.creation.icosahedron()
    sphere.vertices *= radius
    sphere.apply_translation((0, 0, -radius/1.8))
    mesh = mesh.intersection(sphere)
    if stl is not None:
        trimesh2stl(stl, mesh)
    if verbose:
        display_trimesh(mesh)


def create_sponge_cli():
    CLI(create_sponge)


if __name__ == '__main__':
    create_sponge_cli()

