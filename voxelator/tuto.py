import numpy as np
from voxelator.generators import generate_voxel_grid_cylinder, generate_voxel_grid_gyroid
from voxelator.operators import padding
from voxelator.voxel2stl import voxel2stl

grid_size = 51, 51, 101

voxel_cylinder = generate_voxel_grid_cylinder(grid_size=grid_size) * 2.
voxel_gyroid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=2)

voxel_grid = voxel_gyroid * voxel_cylinder

voxel_grid = padding(voxel_grid, padding_size=1, padding_value=20)

voxel_grid = np.abs(voxel_grid)
voxel2stl(stl_filepath="../samples/cylinder.stl", voxel_grid=voxel_grid, level_value=0.5, scale=5/grid_size[0])