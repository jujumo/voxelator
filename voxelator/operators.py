import numpy as np
from typing import Tuple, Union


def padding(
        voxel_grid: np.ndarray,
        padding_size: Union[int, Tuple[int, int]],
        padding_value: float = 0.0
):
    voxel_grid = np.pad(voxel_grid, padding_size, 'constant', constant_values=padding_value)
    return voxel_grid


def sigmoid(
    voxel_grid
):
    return 1. / (1. + np.exp(-voxel_grid))
