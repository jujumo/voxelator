import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track
from voxelator.convertors import voxel2trimesh, mesh2voxel
from voxelator.operators import padding, blur
from voxelator.display import display_trimesh
from jsonargparse import CLI
from typing import Optional
import os.path as path
import trimesh


def voxelize(
    stl: str, npy: Optional[str] = None,
    voxel_size: float = 1.0,
    verbose: bool = False
):
    """
    voxelize turns a stl mesh into a voxel grid.
    Each voxel cell contains distance to nearest surface.
    voxel_size is in STL unit.
    if no nnpy file path is given, its guessed from stl filename.
    """
    if npy is None:
        npy = path.splitext(stl)[0] + '.npy'
    mesh = trimesh.load(stl)
    voxel_grid = mesh2voxel(mesh=mesh, voxel_size=voxel_size)
    np.save(npy, voxel_grid)

    if verbose:
        remesh = voxel2trimesh(voxel_grid, level=0.0)
        display_trimesh(remesh)


def voxelize_cli():
    CLI(voxelize)


if __name__ == '__main__':
    voxelize_cli()
