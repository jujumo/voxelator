import numpy as np
from voxelator.generators import generate_voxel_grid_cylinder, generate_voxel_grid_gyroid
from voxelator.operators import padding, sigmoid
from voxelator.convertors import voxel2trimesh, trimesh2stl
from voxelator.display import display_trimesh
import trimesh
from jsonargparse import CLI
from typing import Optional
import os.path as path
from PIL import Image, ImageOps


def img2stl(
    img_filepath: str,
    stl_filepath: Optional[str] = None,
    level: float = 0.5,
    scale: float = 1.0,
    background: Optional[float] = None,
    verbose: bool = False
):
    if stl_filepath is None:
        stl_filepath = path.splitext(img_filepath)[0] + '.stl'
    img = Image.open(img_filepath)
    img = ImageOps.grayscale(img)
    voxel_grid = np.array(img).astype(float) / 255.
    voxel_grid = np.expand_dims(voxel_grid, axis=2)
    if background is not None:
        background = np.ones_like(voxel_grid) * float(background)
        voxel_grid = 1.0 - voxel_grid
        voxel_grid = np.concatenate([background, voxel_grid], axis=2)
    voxel_grid = padding(voxel_grid, padding_value=1.)
    mesh = voxel2trimesh(voxel_grid, level=level, scale=scale)
    trimesh2stl(stl_filepath, mesh)
    if verbose:
        display_trimesh(mesh)


def npy2stl(
    npy_filepath: str,
    stl_filepath: Optional[str] = None,
    level: float = 0.0,
    scale: float = 1.0,
    verbose: bool = False
):
    if stl_filepath is None:
        stl_filepath = path.splitext(npy_filepath)[0] + '.stl'
    voxel_grid = np.load(npy_filepath)
    voxel_grid = padding(voxel_grid, padding_value=1.)

    mesh = voxel2trimesh(voxel_grid, level=level, scale=scale)
    trimesh2stl(stl_filepath, mesh)
    if verbose:
        display_trimesh(mesh)


def npy2stl_cli():
    CLI(img2stl)


if __name__ == '__main__':
    npy2stl_cli()

