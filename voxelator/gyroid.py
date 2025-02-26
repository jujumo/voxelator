import numpy as np
import os.path as path
from jsonargparse import CLI
from voxelator.voxel2stl import voxel2stl
from voxelator.operators import padding


def gyroid2voxel(
    bound=np.pi * 2,
    nb_samples=80,
):
    r = np.linspace(start=-bound, stop=bound, num=nb_samples)
    x, y, z = np.meshgrid(r, r, r)
    # voxel_grid = np.cos(xx) + np.cos(yy) + np.cos(zz)
    voxel_grid = np.sin(x) * np.cos(y) +\
                 np.sin(y) * np.cos(z) +\
                 np.sin(z) * np.cos(x)

    voxel_grid = np.abs(voxel_grid)
    return voxel_grid


def gyroid2file(
    filepath: str,
    level_value=0.1,
    bound=np.pi * 2,
    nb_samples=200,
    padding_size=10
):
    voxel_grid = gyroid2voxel(bound, nb_samples)
    voxel_grid = padding(voxel_grid, padding_size=padding_size, padding_value=10)
    if path.splitext(filepath)[1].upper() == '.STL':
        voxel2stl(stl_filepath=filepath, voxel_grid=voxel_grid, level_value=level_value)
    else:
        np.save(filepath, voxel_grid)


def create_gyroid_cli():
    CLI(gyroid2file)


if __name__ == '__main__':
    create_gyroid_cli()


