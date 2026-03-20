import numpy as np
from voxelator.operators import padding
from voxelator.convertors import voxel_to_trimesh, trimesh_to_stl, image_file_to_voxel
from voxelator.display import display_trimesh
from jsonargparse import CLI
from typing import Optional
import os.path as path


def arr2stl(
    arr_filepath: str,
    stl_filepath: Optional[str] = None,
    level: float = 0.5,
    scale: float = 1.0,
    background: Optional[float] = 0.1,
    verbose: bool = False
):
    if stl_filepath is None:
        stl_filepath = path.splitext(arr_filepath)[0] + '.stl'
    input_ext = path.splitext(arr_filepath)[1].lower()
    voxel_grid = None
    if input_ext == '.png':
        voxel_grid = image_file_to_voxel(arr_filepath)
    if input_ext == '.npy':
        voxel_grid = np.load(voxel_grid)
    if background is not None:
        background = np.ones_like(voxel_grid) * float(background)
        voxel_grid = np.concatenate([background, voxel_grid], axis=2)

    voxel_grid = padding(voxel_grid, padding_size=1, padding_value=2.0)
    mesh = voxel_to_trimesh(voxel_grid, level=level, scale=(1, 1, scale))
    trimesh_to_stl(stl_filepath, mesh)
    if verbose:
        display_trimesh(mesh)


def arr2stl_cli():
    CLI(arr2stl)


if __name__ == '__main__':
    arr2stl_cli()

