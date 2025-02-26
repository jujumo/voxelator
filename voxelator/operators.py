import numpy as np
import stl
from skimage import measure
from rich.progress import track
from typing import Tuple, Union


def padding(
        voxel_grid: np.ndarray,
        padding_size: Union[int, Tuple[int, int]],
        padding_value: float = 0.0
):
    voxel_grid = np.pad(voxel_grid, padding_size, 'constant', constant_values=padding_value)
    return voxel_grid


def voxel2mesh(
        voxel_grid: np.ndarray,
        level_value: float = 0.0,
        scale: float = 1.0
) -> stl.mesh.Mesh:
    # Use marching cubes to obtain the surface mesh
    vertices, faces, normals, values = measure.marching_cubes(
        voxel_grid,
        level=level_value,
        spacing=(1, 1, 1)
    )

    # apply scale
    vertices = vertices * scale

    voxel_mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in track(enumerate(faces)):
        for j in range(3):
            voxel_mesh.vectors[i][j] = vertices[f[j], :]
    return voxel_mesh
