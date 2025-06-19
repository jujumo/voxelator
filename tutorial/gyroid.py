import matplotlib.pyplot as plt
import numpy as np
from voxelator.generators import generate_voxel_grid_gyroid
from voxelator.operators import padding, mesh_centering
from voxelator.convertors import voxel2trimesh, trimesh2stl
from voxelator.display import display_trimesh
import trimesh
from jsonargparse import CLI
from typing import Optional


def create_gyroid(
    stl: str,
    size: float = 50.,
    periods: float = 3.5,
    thickness: float = 1,
    definition: int = 200,
    verbose: bool = False,
    shape: str = 'cube'
):
    grid_size = np.array([1, 1, 1]) * (definition // 2) * 2
    gyroid_shift = 0, np.pi/2, 0
    scale = size / grid_size[0]
    period_size = size / periods
    iso_surface = 4. * thickness / period_size     # thickness is in mm (same as size)

    voxel_grid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=periods, grid_shifts=gyroid_shift)
    voxel_grid = np.abs(voxel_grid) - iso_surface  # make a thin surface around 0.
    voxel_grid = padding(voxel_grid, padding_value=1.)
    mesh = voxel2trimesh(voxel_grid, level=0.0, scale=scale)
    mesh.apply_translation(-mesh.centroid)

    shape_mesh = None
    if shape == 'pole':
        shape_mesh = trimesh.creation.cylinder(radius=size / 4., height=size, sections=64)
    if shape == 'capsule':
        shape_mesh = trimesh.creation.capsule(radius=size / 4., height=size)
        shape_mesh.apply_translation((0, 0, -size/4.))
    if shape == 'sponge':
        shape_mesh = trimesh.creation.icosphere(subdivisions=3)  # trimesh.creation.icosahedron()
        shape_mesh.vertices *= size / 2.0
        shape_mesh.apply_translation((0, 0, (-3. / 10.) * size))

    if shape_mesh is not None:
        mesh = mesh.intersection(shape_mesh)

    if stl is not None:
        trimesh2stl(stl, mesh)
    if verbose:
        display_trimesh(mesh)


def create_gyroid_cli():
    CLI(create_gyroid)


if __name__ == '__main__':
    create_gyroid_cli()

