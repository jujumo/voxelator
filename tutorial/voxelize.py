import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track
from voxelator.convertors import voxel_to_trimesh, mesh_to_voxel
from voxelator.operators import padding, blur
from voxelator.display import display_trimesh
from jsonargparse import CLI, ArgumentParser
from typing import Optional
import os.path as path
import trimesh


def voxelize(
    stl: Optional[str] = None,
    npy: Optional[str] = None,
    voxel_size: float = 1.0,
    verbosity: int = 0
):
    """
    voxelize turns a stl mesh into a voxel grid.
    Each voxel cell contains distance to nearest surface.
    voxel_size is in STL unit.
    if no nnpy file path is given, its guessed from stl filename.
    """

    if stl is None:
        raise ValueError("stl filename not given")

    if npy is None:
        npy = path.splitext(stl)[0] + '.npy'
    mesh = trimesh.load(stl)
    voxel_grid = mesh_to_voxel(mesh=mesh, voxel_size=voxel_size)
    np.save(npy, voxel_grid)

    if verbosity >= 1:
        remesh = voxel_to_trimesh(voxel_grid, level=0.0)
        display_trimesh(remesh)


class AliasingParser(ArgumentParser):
    def add_argument(self, *args, **kwargs):
        if args == ('--stl',): args += ('-i',)
        if args == ('--npy',): args += ('-o',)
        if args == ('--voxel_size',): args += ('-s',)
        if args == ('--verbosity',): args += ('-v',)
        return super().add_argument(*args, **kwargs)


def voxelize_cli():
    CLI(voxelize, parser_class=AliasingParser)


if __name__ == '__main__':
    voxelize_cli()
