import numpy as np
import scipy
from typing import Tuple, Union


def padding(
        voxel_grid: np.ndarray,
        padding_value: float = 0.0,
        padding_size: Union[int, Tuple[int, int]] = 1
):
    voxel_grid = np.pad(voxel_grid, padding_size, 'constant', constant_values=padding_value)
    return voxel_grid


def sigmoid(
    voxel_grid
):
    return 1. / (1. + np.exp(-voxel_grid))


def blur(
    voxel_grid: np.ndarray,
    size: int = 10
):
    k = np.ones(size) / size
    # Convolve over all three axes in a for loop
    voxel_grid = voxel_grid.copy()
    for i in range(3):
        voxel_grid = scipy.ndimage.convolve1d(voxel_grid, k, axis=i)
    return voxel_grid


def mesh_centering(
        mesh
):
    mesh.apply_translation(-mesh.centroid)  # center mesh to 0.