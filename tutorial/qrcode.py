import matplotlib.pyplot as plt
import numpy as np
from voxelator.convertors import voxel2trimesh, trimesh2stl, img2voxel
from voxelator.operators import padding, blur
from voxelator.display import display_trimesh
from jsonargparse import CLI
from typing import Optional
import os.path as path
import trimesh


def qrcode2stl(
    img_filepath: str,
    stl_black: Optional[str] = None,
    stl_white: Optional[str] = None,
    size: float = -1.0,
    depth: float = 0.6,
    gap: float = 0.2,
    plate_black: bool = False,
    plate_depth: float = 1.4,
    verbose: bool = False
):
    """
    create a physical (stl) qrcode from an image. Assume the QRCode is black over white background.
    size: size o the biggest side in stl unit (mm). If -1, assume image is 96 px/inch
    depth: depth in stl unit (mm)
    gap: size of the gap between white and black mesh in stl unit (mm)
    """
    voxel_grid = img2voxel(img_filepath)
    image_size_voxels = voxel_grid.shape
    if stl_black is None:
        stl_black = path.splitext(img_filepath)[0] + '_w.stl'
    if stl_white is None:
        stl_white = path.splitext(img_filepath)[0] + '_b.stl'
    if size < 0:
        # guess size from image definition assuming 4 pixel / mm
        size = image_size_voxels[0] / 4.

    # compute voxel / mm scale
    if image_size_voxels[0] >= image_size_voxels[1]:
        voxel_per_mm = image_size_voxels[0]/size
    if image_size_voxels[0] < image_size_voxels[1]:
        voxel_per_mm = image_size_voxels[1]/size
    # compute depth in voxels
    depth_voxel = int(np.ceil(depth * voxel_per_mm))
    depth_voxel = depth_voxel - depth_voxel % 2  # make it even number
    if depth_voxel < 1:
        raise ValueError('depth should be positive')
    gap_voxel = int(np.ceil(gap * voxel_per_mm))
    voxel_grid = np.concatenate([voxel_grid] * depth_voxel, axis=2)
    if plate_depth > 0:
        # 1 = white 0 = black
        plate = np.empty((image_size_voxels[0], image_size_voxels[1], int(plate_depth*voxel_per_mm)))
        plate.fill(0. if plate_black else 1.0)
        voxel_grid = np.concatenate([plate, voxel_grid], axis=2)
    voxel_grid = blur(voxel_grid=voxel_grid, size=gap_voxel * 2)

    voxel_grid_black = padding(1.0 - voxel_grid, padding_size=1, padding_value=1.0)
    mesh_black = voxel2trimesh(voxel_grid_black, level=0.25)
    voxel_grid_white = padding(voxel_grid, padding_size=1, padding_value=1.0)
    mesh_white = voxel2trimesh(voxel_grid_white, level=0.25)
    trimesh2stl(stl_black, mesh_black, scale=1./voxel_per_mm)
    trimesh2stl(stl_white, mesh_white, scale=1./voxel_per_mm)
    if verbose:
        all_mesh = trimesh.util.concatenate([mesh_black, mesh_white])
        display_trimesh(all_mesh)


def qrcode_cli():
    CLI(qrcode2stl)


if __name__ == '__main__':
    qrcode_cli()

