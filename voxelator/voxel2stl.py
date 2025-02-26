import numpy as np
from stl import mesh
from skimage import measure
import os.path as path
from rich.progress import track
from typing import Optional, Tuple
from jsonargparse import CLI



def voxel2mesh(
        voxel_grid: np.ndarray,
        level_value: float = 0.0,
        padding: Optional[Tuple[int, float]] = None
) -> mesh.Mesh:
    if padding:
        padding_size, padding_value = padding[0], padding[1]
        voxel_grid = np.pad(voxel_grid, padding[0], 'constant', constant_values=padding_value)

    # Use marching cubes to obtain the surface mesh
    vertices, faces, normals, values = measure.marching_cubes(
        voxel_grid,
        level=level_value,
        spacing=(1, 1, 1)
    )
    maillage_stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in track(enumerate(faces)):
        for j in range(3):
            maillage_stl.vectors[i][j] = vertices[f[j], :]
    return maillage_stl


def voxel2stl(
        stl_filepath: str,
        voxel_grid: np.ndarray,
        level_value: float = 1.0,
        padding: Optional[Tuple[int, float]] = None
):
    stl_mesh = voxel2mesh(voxel_grid=voxel_grid, level_value=level_value, padding=padding)
    stl_mesh.save(stl_filepath)


def voxel2stl_cli():
    CLI(voxel2stl)


if __name__ == '__main__':
    voxel2stl_cli()
