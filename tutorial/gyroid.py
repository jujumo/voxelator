import matplotlib.pyplot as plt
import numpy as np
from voxelator.generators import generate_voxel_grid_gyroid
from voxelator.operators import padding, mesh_centering
from voxelator.convertors import voxel_to_trimesh, trimesh_to_stl
from voxelator.display import display_trimesh
import trimesh
from jsonargparse import CLI, ArgumentParser
from typing import Optional


def create_gyroid(
    stl: Optional[str] = None,
    size: float = 50.,
    periods: float = 3.5,
    thickness: float = 1,
    definition: int = 200,
    invert: bool = False,
    shape: str = 'cube',
    verbosity: int = 0,
):
    grid_size = np.array([1, 1, 1]) * (definition // 2) * 2
    gyroid_shift = 0, np.pi/2, 0
    scale = size / grid_size[0]
    period_size = size / periods
    iso_surface = 4. * thickness / period_size     # thickness is in mm (same as size)

    voxel_grid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=periods, grid_shifts=gyroid_shift)
    voxel_grid = np.abs(voxel_grid) - iso_surface  # make a thin surface around 0.
    if invert:
        voxel_grid *= -1.0
    voxel_grid = padding(voxel_grid, padding_value=1.)
    mesh = voxel_to_trimesh(voxel_grid, level=0.0, scale=scale)
    mesh.apply_translation(-mesh.centroid)

    shape_mesh = None
    if shape == 'pole':
        shape_mesh = trimesh.creation.cylinder(radius=size / 4., height=size, sections=64)
    if shape == 'capsule':
        shape_mesh = trimesh.creation.capsule(radius=size / 4., height=size)
        shape_mesh.apply_translation((0, 0, -size/4.))
    if shape == 'monolith':
        ratio = 0.5
        shape_mesh = trimesh.creation.box(extents=(size * ratio, size*ratio, size))
    if shape == 'sponge':
        shape_mesh = trimesh.creation.icosphere(subdivisions=3)  # trimesh.creation.icosahedron()
        shape_mesh.vertices *= size / 2.0
        shape_mesh.apply_translation((0, 0, (-3. / 10.) * size))

    if shape_mesh is not None:
        mesh = mesh.intersection(shape_mesh)

    if stl is not None:
        trimesh_to_stl(stl, mesh)
    if verbosity >= 1:
        display_trimesh(mesh)


class AliasingParser(ArgumentParser):
    def add_argument(self, *args, **kwargs):
        if args == ('--stl',): args += ('-i',)
        if args == ('--size',): args += ('-s',)
        if args == ('--periods',): args += ('-p',)
        if args == ('--thickness',): args += ('-t',)
        if args == ('--definition',): args += ('-d',)
        if args == ('--invert',): args += ('-x',)
        if args == ('--verbosity',): args += ('-v',)
        return super().add_argument(*args, **kwargs)


def create_gyroid_cli():
    CLI(create_gyroid, parser_class=AliasingParser)


if __name__ == '__main__':
    create_gyroid_cli()

