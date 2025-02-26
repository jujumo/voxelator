import numpy as np
from voxelator.generators import generate_voxel_grid_cylinder, generate_voxel_grid_gyroid, generate_voxel_grid_sphere
from voxelator.operators import padding
from voxelator.voxel2stl import voxel2stl

grid_size = 101, 101, 201
shift = np.pi/2, 0, 0

voxel_cylinder = generate_voxel_grid_cylinder(grid_size=grid_size) * 1.5
voxel_sphere = 2.0 * generate_voxel_grid_sphere(grid_size=grid_size)
voxel_gyroid = generate_voxel_grid_gyroid(grid_size=grid_size, grid_periods=2.5, grid_shifts=shift)

voxel_grid = voxel_gyroid + voxel_cylinder

voxel_grid = padding(voxel_grid, padding_size=1, padding_value=20)
# voxel_grid = np.abs(voxel_grid)


voxel2stl(stl_filepath="../samples/cylinder.stl", voxel_grid=voxel_grid, level_value=0.2, scale=40./grid_size[0])