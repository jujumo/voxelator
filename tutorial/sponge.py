import numpy as np
from voxelator.generators import generate_voxel_grid_cylinder, generate_voxel_grid_gyroid
from voxelator.operators import padding, mesh_centering
from voxelator.convertors import voxel_to_trimesh, trimesh_to_stl
from voxelator.display import display_trimesh
import trimesh
from jsonargparse import CLI, ArgumentParser
from typing import Optional


def create_sponge(
    stl: Optional[str] = None,
    periods: float = 3.0,
    radius: float = 50.,
    definition: int = 100,
    thickness: float = 0.3,
    verbosity: int = 0
):
    grid_size = np.array([1, 1, 1]) * definition + 1
    gyroid_shift = 0, np.pi/2, 0
    scale = 2 * radius / grid_size[0]

    voxel_grid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=periods, grid_shifts=gyroid_shift)
    voxel_grid = np.abs(voxel_grid) - (thickness / 2.)
    voxel_grid = padding(voxel_grid, padding_value=1.)
    mesh = voxel_to_trimesh(voxel_grid, level=0.0, scale=scale)
    mesh_centering(mesh)

    sphere = trimesh.creation.icosphere(subdivisions=1)  # trimesh.creation.icosahedron()
    sphere.vertices *= radius
    sphere.apply_translation((0, 0, -radius/1.8))
    mesh = mesh.intersection(sphere)
    if stl is not None:
        trimesh_to_stl(stl, mesh)
    if verbosity >= 1:
        display_trimesh(mesh)


class AliasingParser(ArgumentParser):
    def add_argument(self, *args, **kwargs):
        if args == ('--periods',): args += ('-p',)
        if args == ('--radius',): args += ('-r',)
        if args == ('--definition',): args += ('-d',)
        if args == ('--thickness',): args += ('-t',)
        if args == ('--stl',): args += ('-o',)
        if args == ('--verbosity',): args += ('-v',)
        return super().add_argument(*args, **kwargs)


def create_sponge_cli():
    CLI(create_sponge, parser_class=AliasingParser)


if __name__ == '__main__':
    create_sponge_cli()

