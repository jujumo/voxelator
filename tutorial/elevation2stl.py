import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI, ArgumentParser
from typing import Optional
import os.path as path
import trimesh
from PIL import Image, ImageOps
from voxelator.convertors import elevation_to_voxel, voxel_to_trimesh
from voxelator.convertors import trimesh_to_stl
from voxelator.operators import padding, blur
from voxelator.display import display_trimesh


def elevation2stl(
    elevation_filepath: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    depth: Optional[float] = None,
    stl_filepath: Optional[str] = None,
    verbosity: int = 0
):
    """
    create a physical (stl) qrcode from an image. Assume the QRCode is black over white background.
    size: size o the biggest side in stl unit (mm). If -1, assume image is 96 px/inch
    depth: depth in stl unit (mm)
    gap: size of the gap between white and black mesh in stl unit (mm)
    """
    if elevation_filepath is None:
        raise ValueError('elevation_filepath must be specified.')

    elevation = Image.open(elevation_filepath)
    elevation = ImageOps.grayscale(elevation)
    elevation = np.array(elevation).astype(np.uint8)
    elevation_max_val = np.iinfo(elevation.dtype).max
    voxel_grid = elevation_to_voxel(elevation, add_padding=False)

    voxel_grid = padding(voxel_grid, padding_value=elevation_max_val, padding_size=1)
    voxel_grid_shape = np.array(elevation.shape + tuple([elevation_max_val]))
    scale = np.ones(3)
    if width is not None:
        scale[0] = width / voxel_grid_shape[0]
    if height is not None:
        scale[1] = height / voxel_grid_shape[1]
    if depth is not None:
        scale[2] = depth / voxel_grid_shape[2]
    mesh = voxel_to_trimesh(voxel_grid, scale=scale, level=0)

    if stl_filepath is not None:
        if verbosity >= 1:
            print(f'writing stl to "{stl_filepath}".')
        trimesh_to_stl(stl_filepath=stl_filepath, mesh=mesh)
    if verbosity >= 2:
        mesh.show()


class AliasingParser(ArgumentParser):
    def add_argument(self, *args, **kwargs):
        if args == ('--elevation_filepath',): args += ('-i',)
        if args == ('--stl_filepath',): args += ('-o',)
        if args == ('--width',): args += ('-x',)
        if args == ('--height',): args += ('-y',)
        if args == ('--depth',): args += ('-z',)
        if args == ('--verbosity',): args += ('-v',)
        return super().add_argument(*args, **kwargs)


def elevation2stl_cli():
    CLI(elevation2stl, parser_class=AliasingParser)


if __name__ == '__main__':
    elevation2stl_cli()

