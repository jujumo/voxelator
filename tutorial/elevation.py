import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI, ArgumentParser
from typing import Optional
import os.path as path
import trimesh
from PIL import Image, ImageOps
from voxelator.convertors import elevation_to_voxel, voxel_to_trimesh
from voxelator.operators import padding, blur
from voxelator.display import display_trimesh


def elevation2stl(
    elevation_filepath: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    depth: Optional[float] = None,
    verbosity: int = 0
):
    """
    create a physical (stl) qrcode from an image. Assume the QRCode is black over white background.
    size: size o the biggest side in stl unit (mm). If -1, assume image is 96 px/inch
    depth: depth in stl unit (mm)
    gap: size of the gap between white and black mesh in stl unit (mm)
    """
    elevation = Image.open(elevation_filepath)
    elevation = ImageOps.grayscale(elevation)
    elevation = np.array(elevation)
    elevation = np.clip(elevation, 10, 255)
    plt.imshow(elevation); plt.show()
    # plt.plot(elevation[100]); plt.show()
    voxel_grid = elevation_to_voxel(elevation)
    voxel_grid = padding(voxel_grid, padding_value=1.0)
    mesh = voxel_to_trimesh(voxel_grid)
    mesh.show()


class AliasingParser(ArgumentParser):
    def add_argument(self, *args, **kwargs):
        if args == ('--elevation_filepath',): args += ('-i',)
        if args == ('--verbosity',): args += ('-v',)
        return super().add_argument(*args, **kwargs)

def elevation2stl_cli():
    CLI(elevation2stl, parser_class=AliasingParser)


if __name__ == '__main__':
    elevation2stl_cli()

